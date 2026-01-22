import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
class TeXDimProc:
    """Helper class for for finding the size of TeX snippets

    Uses preview.sty
    """

    def __init__(self, template, options):
        self.template = template
        self.snippets_code = []
        self.snippets_id = []
        self.options = options
        self.dimext_re = re.compile(dimext, re.MULTILINE | re.VERBOSE)
        pass

    def add_snippet(self, snippet_id, code):
        """A a snippet of code to be processed"""
        self.snippets_id.append(snippet_id)
        self.snippets_code.append(code)

    def process(self):
        """Process all snippets of code with TeX and preview.sty

        Results are stored in the texdimlist and texdims class attributes.
        Returns False if preprocessing fails
        """
        import shutil
        if len(self.snippets_code) == 0:
            log.warning('No labels to preprocess')
            return True
        self.tempdir = tempfile.mkdtemp(prefix='dot2tex')
        log.debug('Creating temporary directroy %s' % self.tempdir)
        self.tempfilename = os.path.join(self.tempdir, 'dot2tex.tex')
        log.debug('Creating temporary file %s' % self.tempfilename)
        s = ''
        for n in self.snippets_code:
            s += '\\begin{preview}%\n'
            s += n.strip() + '%\n'
            s += '\\end{preview}%\n'
        with open(self.tempfilename, 'w') as f:
            f.write(self.template.replace('<<preproccode>>', s))
        with open(self.tempfilename, 'r') as f:
            s = f.read()
        log.debug('Code written to %s\n' % self.tempfilename + s)
        self.parse_log_file()
        shutil.rmtree(self.tempdir)
        log.debug('Temporary directory and files deleted')
        if self.texdims:
            return True
        else:
            return False

    def parse_log_file(self):
        logfilename = os.path.splitext(self.tempfilename)[0] + '.log'
        tmpdir = os.getcwd()
        os.chdir(os.path.split(logfilename)[0])
        if self.options.get('usepdflatex'):
            command = 'pdflatex -interaction=nonstopmode %s' % self.tempfilename
        else:
            command = 'latex -interaction=nonstopmode %s' % self.tempfilename
        log.debug('Running command: %s' % command)
        p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE, close_fds=sys.platform != 'win32')
        stdout, stderr = (p.stdout, p.stderr)
        try:
            data = stdout.read()
            log.debug('stdout from latex\n %s', data)
        finally:
            stdout.close()
        try:
            error_data = stderr.read()
            if error_data:
                log.debug('latex STDERR %s', error_data)
        finally:
            stderr.close()
        p.kill()
        p.wait()
        with open(logfilename, 'r') as f:
            logdata = f.read()
        log.debug('Logfile from LaTeX run: \n' + logdata)
        os.chdir(tmpdir)
        texdimdata = self.dimext_re.findall(logdata)
        log.debug('Texdimdata: ' + str(texdimdata))
        if len(texdimdata) == 0:
            log.error('No dimension data could be extracted from dot2tex.tex.')
            self.texdims = None
            return
        c = 1.0 / 4736286
        self.texdims = {}
        self.texdimlist = [(float(i[1]) * c, float(i[2]) * c, float(i[3]) * c) for i in texdimdata]
        self.texdims = dict(zip(self.snippets_id, self.texdimlist))