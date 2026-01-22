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
class IPythonDirective(Directive):
    has_content = True
    required_arguments = 0
    optional_arguments = 4
    final_argumuent_whitespace = True
    option_spec = {'python': directives.unchanged, 'suppress': directives.flag, 'verbatim': directives.flag, 'doctest': directives.flag, 'okexcept': directives.flag, 'okwarning': directives.flag}
    shell = None
    seen_docs = set()

    def get_config_options(self):
        config = self.state.document.settings.env.config
        savefig_dir = config.ipython_savefig_dir
        source_dir = self.state.document.settings.env.srcdir
        savefig_dir = os.path.join(source_dir, savefig_dir)
        rgxin = config.ipython_rgxin
        rgxout = config.ipython_rgxout
        warning_is_error = config.ipython_warning_is_error
        promptin = config.ipython_promptin
        promptout = config.ipython_promptout
        mplbackend = config.ipython_mplbackend
        exec_lines = config.ipython_execlines
        hold_count = config.ipython_holdcount
        return (savefig_dir, source_dir, rgxin, rgxout, promptin, promptout, mplbackend, exec_lines, hold_count, warning_is_error)

    def setup(self):
        savefig_dir, source_dir, rgxin, rgxout, promptin, promptout, mplbackend, exec_lines, hold_count, warning_is_error = self.get_config_options()
        try:
            os.makedirs(savefig_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        if self.shell is None:
            if mplbackend and 'matplotlib.backends' not in sys.modules and use_matplotlib:
                import matplotlib
                matplotlib.use(mplbackend)
            self.shell = EmbeddedSphinxShell(exec_lines)
            self.shell.directive = self
        if not self.state.document.current_source in self.seen_docs:
            self.shell.IP.history_manager.reset()
            self.shell.IP.execution_count = 1
            self.seen_docs.add(self.state.document.current_source)
        self.shell.rgxin = rgxin
        self.shell.rgxout = rgxout
        self.shell.promptin = promptin
        self.shell.promptout = promptout
        self.shell.savefig_dir = savefig_dir
        self.shell.source_dir = source_dir
        self.shell.hold_count = hold_count
        self.shell.warning_is_error = warning_is_error
        self.shell.process_input_line('bookmark ipy_savedir "%s"' % savefig_dir, store_history=False)
        self.shell.clear_cout()
        return (rgxin, rgxout, promptin, promptout)

    def teardown(self):
        self.shell.process_input_line('bookmark -d ipy_savedir', store_history=False)
        self.shell.clear_cout()

    def run(self):
        debug = False
        rgxin, rgxout, promptin, promptout = self.setup()
        options = self.options
        self.shell.is_suppress = 'suppress' in options
        self.shell.is_doctest = 'doctest' in options
        self.shell.is_verbatim = 'verbatim' in options
        self.shell.is_okexcept = 'okexcept' in options
        self.shell.is_okwarning = 'okwarning' in options
        if 'python' in self.arguments:
            content = self.content
            self.content = self.shell.process_pure_python(content)
        parts = '\n'.join(self.content).split('\n\n')
        lines = ['.. code-block:: ipython', '']
        figures = []
        logger = logging.getLogger(__name__)
        for part in parts:
            block = block_parser(part, rgxin, rgxout, promptin, promptout)
            if len(block):
                rows, figure = self.shell.process_block(block)
                for row in rows:
                    lines.extend(['   {0}'.format(line) for line in row.split('\n')])
                if figure is not None:
                    figures.append(figure)
            else:
                message = 'Code input with no code at {}, line {}'.format(self.state.document.current_source, self.state.document.current_line)
                if self.shell.warning_is_error:
                    raise RuntimeError(message)
                else:
                    logger.warning(message)
        for figure in figures:
            lines.append('')
            lines.extend(figure.split('\n'))
            lines.append('')
        if len(lines) > 2:
            if debug:
                print('\n'.join(lines))
            else:
                self.state_machine.insert_input(lines, self.state_machine.input_lines.source(0))
        self.teardown()
        return []