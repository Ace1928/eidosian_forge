import atexit
import errno
import os
import pathlib
import re
import sys
import tempfile
import ast
import warnings
import shutil
from io import StringIO
from docutils.parsers.rst import directives
from docutils.parsers.rst import Directive
from sphinx.util import logging
from traitlets.config import Config
from IPython import InteractiveShell
from IPython.core.profiledir import ProfileDir
class EmbeddedSphinxShell(object):
    """An embedded IPython instance to run inside Sphinx"""

    def __init__(self, exec_lines=None):
        self.cout = StringIO()
        if exec_lines is None:
            exec_lines = []
        config = Config()
        config.HistoryManager.hist_file = ':memory:'
        config.InteractiveShell.autocall = False
        config.InteractiveShell.autoindent = False
        config.InteractiveShell.colors = 'NoColor'
        tmp_profile_dir = tempfile.mkdtemp(prefix='profile_')
        profname = 'auto_profile_sphinx_build'
        pdir = os.path.join(tmp_profile_dir, profname)
        profile = ProfileDir.create_profile_dir(pdir)
        IP = InteractiveShell.instance(config=config, profile_dir=profile)
        atexit.register(self.cleanup)
        self.IP = IP
        self.user_ns = self.IP.user_ns
        self.user_global_ns = self.IP.user_global_ns
        self.input = ''
        self.output = ''
        self.tmp_profile_dir = tmp_profile_dir
        self.is_verbatim = False
        self.is_doctest = False
        self.is_suppress = False
        self.directive = None
        self._pyplot_imported = False
        for line in exec_lines:
            self.process_input_line(line, store_history=False)

    def cleanup(self):
        shutil.rmtree(self.tmp_profile_dir, ignore_errors=True)

    def clear_cout(self):
        self.cout.seek(0)
        self.cout.truncate(0)

    def process_input_line(self, line, store_history):
        return self.process_input_lines([line], store_history=store_history)

    def process_input_lines(self, lines, store_history=True):
        """process the input, capturing stdout"""
        stdout = sys.stdout
        source_raw = '\n'.join(lines)
        try:
            sys.stdout = self.cout
            self.IP.run_cell(source_raw, store_history=store_history)
        finally:
            sys.stdout = stdout

    def process_image(self, decorator):
        """
        # build out an image directive like
        # .. image:: somefile.png
        #    :width 4in
        #
        # from an input like
        # savefig somefile.png width=4in
        """
        savefig_dir = self.savefig_dir
        source_dir = self.source_dir
        saveargs = decorator.split(' ')
        filename = saveargs[1]
        path = pathlib.Path(savefig_dir, filename)
        outfile = '/' + path.relative_to(source_dir).as_posix()
        imagerows = ['.. image:: %s' % outfile]
        for kwarg in saveargs[2:]:
            arg, val = kwarg.split('=')
            arg = arg.strip()
            val = val.strip()
            imagerows.append('   :%s: %s' % (arg, val))
        image_file = os.path.basename(outfile)
        image_directive = '\n'.join(imagerows)
        return (image_file, image_directive)

    def process_input(self, data, input_prompt, lineno):
        """
        Process data block for INPUT token.

        """
        decorator, input, rest = data
        image_file = None
        image_directive = None
        is_verbatim = decorator == '@verbatim' or self.is_verbatim
        is_doctest = decorator is not None and decorator.startswith('@doctest') or self.is_doctest
        is_suppress = decorator == '@suppress' or self.is_suppress
        is_okexcept = decorator == '@okexcept' or self.is_okexcept
        is_okwarning = decorator == '@okwarning' or self.is_okwarning
        is_savefig = decorator is not None and decorator.startswith('@savefig')
        input_lines = input.split('\n')
        if len(input_lines) > 1:
            if input_lines[-1] != '':
                input_lines.append('')
        continuation = '   %s:' % ''.join(['.'] * (len(str(lineno)) + 2))
        if is_savefig:
            image_file, image_directive = self.process_image(decorator)
        ret = []
        is_semicolon = False
        if is_suppress and self.hold_count:
            store_history = False
        else:
            store_history = True
        with warnings.catch_warnings(record=True) as ws:
            if input_lines[0].endswith(';'):
                is_semicolon = True
            if is_verbatim:
                self.process_input_lines([''])
                self.IP.execution_count += 1
            else:
                self.process_input_lines(input_lines, store_history=store_history)
        if not is_suppress:
            for i, line in enumerate(input_lines):
                if i == 0:
                    formatted_line = '%s %s' % (input_prompt, line)
                else:
                    formatted_line = '%s %s' % (continuation, line)
                ret.append(formatted_line)
        if not is_suppress and len(rest.strip()) and is_verbatim:
            ret.append(rest)
        self.cout.seek(0)
        processed_output = self.cout.read()
        if not is_suppress and (not is_semicolon):
            ret.append(processed_output)
        elif is_semicolon:
            ret.append('')
        filename = 'Unknown'
        lineno = 0
        if self.directive.state:
            filename = self.directive.state.document.current_source
            lineno = self.directive.state.document.current_line
        logger = logging.getLogger(__name__)
        if not is_okexcept and ('Traceback' in processed_output or 'SyntaxError' in processed_output):
            s = '\n>>>' + '-' * 73 + '\n'
            s += 'Exception in %s at block ending on line %s\n' % (filename, lineno)
            s += 'Specify :okexcept: as an option in the ipython:: block to suppress this message\n'
            s += processed_output + '\n'
            s += '<<<' + '-' * 73
            logger.warning(s)
            if self.warning_is_error:
                raise RuntimeError('Unexpected exception in `{}` line {}'.format(filename, lineno))
        if not is_okwarning:
            for w in ws:
                s = '\n>>>' + '-' * 73 + '\n'
                s += 'Warning in %s at block ending on line %s\n' % (filename, lineno)
                s += 'Specify :okwarning: as an option in the ipython:: block to suppress this message\n'
                s += '-' * 76 + '\n'
                s += warnings.formatwarning(w.message, w.category, w.filename, w.lineno, w.line)
                s += '<<<' + '-' * 73
                logger.warning(s)
                if self.warning_is_error:
                    raise RuntimeError('Unexpected warning in `{}` line {}'.format(filename, lineno))
        self.clear_cout()
        return (ret, input_lines, processed_output, is_doctest, decorator, image_file, image_directive)

    def process_output(self, data, output_prompt, input_lines, output, is_doctest, decorator, image_file):
        """
        Process data block for OUTPUT token.

        """
        TAB = ' ' * 4
        if is_doctest and output is not None:
            found = output
            found = found.strip()
            submitted = data.strip()
            if self.directive is None:
                source = 'Unavailable'
                content = 'Unavailable'
            else:
                source = self.directive.state.document.current_source
                content = self.directive.content
                content = '\n'.join([TAB + line for line in content])
            ind = found.find(output_prompt)
            if ind < 0:
                e = 'output does not contain output prompt\n\nDocument source: {0}\n\nRaw content: \n{1}\n\nInput line(s):\n{TAB}{2}\n\nOutput line(s):\n{TAB}{3}\n\n'
                e = e.format(source, content, '\n'.join(input_lines), repr(found), TAB=TAB)
                raise RuntimeError(e)
            found = found[len(output_prompt):].strip()
            if decorator.strip() == '@doctest':
                if found != submitted:
                    e = 'doctest failure\n\nDocument source: {0}\n\nRaw content: \n{1}\n\nOn input line(s):\n{TAB}{2}\n\nwe found output:\n{TAB}{3}\n\ninstead of the expected:\n{TAB}{4}\n\n'
                    e = e.format(source, content, '\n'.join(input_lines), repr(found), repr(submitted), TAB=TAB)
                    raise RuntimeError(e)
            else:
                self.custom_doctest(decorator, input_lines, found, submitted)
        out_data = []
        is_verbatim = decorator == '@verbatim' or self.is_verbatim
        if is_verbatim and data.strip():
            out_data.append('{0} {1}\n'.format(output_prompt, data))
        return out_data

    def process_comment(self, data):
        """Process data fPblock for COMMENT token."""
        if not self.is_suppress:
            return [data]

    def save_image(self, image_file):
        """
        Saves the image file to disk.
        """
        self.ensure_pyplot()
        command = 'plt.gcf().savefig("%s")' % image_file
        self.process_input_line('bookmark ipy_thisdir', store_history=False)
        self.process_input_line('cd -b ipy_savedir', store_history=False)
        self.process_input_line(command, store_history=False)
        self.process_input_line('cd -b ipy_thisdir', store_history=False)
        self.process_input_line('bookmark -d ipy_thisdir', store_history=False)
        self.clear_cout()

    def process_block(self, block):
        """
        process block from the block_parser and return a list of processed lines
        """
        ret = []
        output = None
        input_lines = None
        lineno = self.IP.execution_count
        input_prompt = self.promptin % lineno
        output_prompt = self.promptout % lineno
        image_file = None
        image_directive = None
        found_input = False
        for token, data in block:
            if token == COMMENT:
                out_data = self.process_comment(data)
            elif token == INPUT:
                found_input = True
                out_data, input_lines, output, is_doctest, decorator, image_file, image_directive = self.process_input(data, input_prompt, lineno)
            elif token == OUTPUT:
                if not found_input:
                    TAB = ' ' * 4
                    linenumber = 0
                    source = 'Unavailable'
                    content = 'Unavailable'
                    if self.directive:
                        linenumber = self.directive.state.document.current_line
                        source = self.directive.state.document.current_source
                        content = self.directive.content
                        content = '\n'.join([TAB + line for line in content])
                    e = '\n\nInvalid block: Block contains an output prompt without an input prompt.\n\nDocument source: {0}\n\nContent begins at line {1}: \n\n{2}\n\nProblematic block within content: \n\n{TAB}{3}\n\n'
                    e = e.format(source, linenumber, content, block, TAB=TAB)
                    sys.stdout.write(e)
                    raise RuntimeError('An invalid block was detected.')
                out_data = self.process_output(data, output_prompt, input_lines, output, is_doctest, decorator, image_file)
                if out_data:
                    assert ret[-1] == ''
                    del ret[-1]
            if out_data:
                ret.extend(out_data)
        if image_file is not None:
            self.save_image(image_file)
        return (ret, image_directive)

    def ensure_pyplot(self):
        """
        Ensures that pyplot has been imported into the embedded IPython shell.

        Also, makes sure to set the backend appropriately if not set already.

        """
        if not self._pyplot_imported:
            if 'matplotlib.backends' not in sys.modules:
                import matplotlib
                matplotlib.use('agg')
            self.process_input_line('import matplotlib.pyplot as plt', store_history=False)
            self._pyplot_imported = True

    def process_pure_python(self, content):
        """
        content is a list of strings. it is unedited directive content

        This runs it line by line in the InteractiveShell, prepends
        prompts as needed capturing stderr and stdout, then returns
        the content as a list as if it were ipython code
        """
        output = []
        savefig = False
        multiline = False
        multiline_start = None
        fmtin = self.promptin
        ct = 0
        for lineno, line in enumerate(content):
            line_stripped = line.strip()
            if not len(line):
                output.append(line)
                continue
            if any((line_stripped.startswith('@' + pseudo_decorator) for pseudo_decorator in PSEUDO_DECORATORS)):
                output.extend([line])
                if 'savefig' in line:
                    savefig = True
                continue
            if line_stripped.startswith('#'):
                output.extend([line])
                continue
            continuation = u'   %s:' % ''.join(['.'] * (len(str(ct)) + 2))
            if not multiline:
                modified = u'%s %s' % (fmtin % ct, line_stripped)
                output.append(modified)
                ct += 1
                try:
                    ast.parse(line_stripped)
                    output.append(u'')
                except Exception:
                    multiline = True
                    multiline_start = lineno
            else:
                modified = u'%s %s' % (continuation, line)
                output.append(modified)
                if len(content) > lineno + 1:
                    nextline = content[lineno + 1]
                    if len(nextline) - len(nextline.lstrip()) > 3:
                        continue
                try:
                    mod = ast.parse('\n'.join(content[multiline_start:lineno + 1]))
                    if isinstance(mod.body[0], ast.FunctionDef):
                        for element in mod.body[0].body:
                            if isinstance(element, ast.Return):
                                multiline = False
                    else:
                        output.append(u'')
                        multiline = False
                except Exception:
                    pass
            if savefig:
                self.ensure_pyplot()
                self.process_input_line('plt.clf()', store_history=False)
                self.clear_cout()
                savefig = False
        return output

    def custom_doctest(self, decorator, input_lines, found, submitted):
        """
        Perform a specialized doctest.

        """
        from .custom_doctests import doctests
        args = decorator.split()
        doctest_type = args[1]
        if doctest_type in doctests:
            doctests[doctest_type](self, args, input_lines, found, submitted)
        else:
            e = 'Invalid option to @doctest: {0}'.format(doctest_type)
            raise Exception(e)