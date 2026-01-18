import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_unicode_commit_message_is_filename(self):
    """Unicode commit message same as a filename (Bug #563646).
        """
    self.requireFeature(features.UnicodeFilenameFeature)
    file_name = '€'
    self.run_bzr(['init'])
    with open(file_name, 'w') as f:
        f.write('hello world')
    self.run_bzr(['add'])
    out, err = self.run_bzr(['commit', '-m', file_name])
    reflags = re.MULTILINE | re.DOTALL | re.UNICODE
    te = osutils.get_terminal_encoding()
    self.assertContainsRe(err, 'The commit message is a file name:', flags=reflags)
    default_get_terminal_enc = osutils.get_terminal_encoding
    try:
        osutils.get_terminal_encoding = lambda trace=None: 'ascii'
        file_name = 'fooሴ'
        with open(file_name, 'w') as f:
            f.write('hello world')
        self.run_bzr(['add'])
        out, err = self.run_bzr(['commit', '-m', file_name])
        reflags = re.MULTILINE | re.DOTALL | re.UNICODE
        te = osutils.get_terminal_encoding()
        self.assertContainsRe(err, 'The commit message is a file name:', flags=reflags)
    finally:
        osutils.get_terminal_encoding = default_get_terminal_enc