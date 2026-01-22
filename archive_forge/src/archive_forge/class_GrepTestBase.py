import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
class GrepTestBase(tests.TestCaseWithTransport):
    """Base class for testing grep.

    Provides support methods for creating directory and file revisions.
    """
    _reflags = re.MULTILINE | re.DOTALL

    def _mk_file(self, path, line_prefix, total_lines, versioned):
        text = ''
        for i in range(total_lines):
            text += line_prefix + str(i + 1) + '\n'
        with open(path, 'w') as f:
            f.write(text)
        if versioned:
            self.run_bzr(['add', path])
            self.run_bzr(['ci', '-m', '"' + path + '"'])

    def _update_file(self, path, text, checkin=True):
        """append text to file 'path' and check it in"""
        with open(path, 'a') as f:
            f.write(text)
        if checkin:
            self.run_bzr(['ci', path, '-m', '"' + path + '"'])

    def _mk_unknown_file(self, path, line_prefix='line', total_lines=10):
        self._mk_file(path, line_prefix, total_lines, versioned=False)

    def _mk_versioned_file(self, path, line_prefix='line', total_lines=10):
        self._mk_file(path, line_prefix, total_lines, versioned=True)

    def _mk_dir(self, path, versioned):
        os.mkdir(path)
        if versioned:
            self.run_bzr(['add', path])
            self.run_bzr(['ci', '-m', '"' + path + '"'])

    def _mk_unknown_dir(self, path):
        self._mk_dir(path, versioned=False)

    def _mk_versioned_dir(self, path):
        self._mk_dir(path, versioned=True)