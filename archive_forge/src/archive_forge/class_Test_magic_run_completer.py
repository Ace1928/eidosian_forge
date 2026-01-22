import os
import shutil
import sys
import tempfile
import unittest
from os.path import join
from tempfile import TemporaryDirectory
from IPython.core.completerlib import magic_run_completer, module_completion, try_import
from IPython.testing.decorators import onlyif_unicode_paths
class Test_magic_run_completer(unittest.TestCase):
    files = [u'aao.py', u'a.py', u'b.py', u'aao.txt']
    dirs = [u'adir/', 'bdir/']

    def setUp(self):
        self.BASETESTDIR = tempfile.mkdtemp()
        for fil in self.files:
            with open(join(self.BASETESTDIR, fil), 'w', encoding='utf-8') as sfile:
                sfile.write('pass\n')
        for d in self.dirs:
            os.mkdir(join(self.BASETESTDIR, d))
        self.oldpath = os.getcwd()
        os.chdir(self.BASETESTDIR)

    def tearDown(self):
        os.chdir(self.oldpath)
        shutil.rmtree(self.BASETESTDIR)

    def test_1(self):
        """Test magic_run_completer, should match two alternatives
        """
        event = MockEvent(u'%run a')
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u'a.py', u'aao.py', u'adir/'})

    def test_2(self):
        """Test magic_run_completer, should match one alternative
        """
        event = MockEvent(u'%run aa')
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u'aao.py'})

    def test_3(self):
        """Test magic_run_completer with unterminated " """
        event = MockEvent(u'%run "a')
        mockself = None
        match = set(magic_run_completer(mockself, event))
        self.assertEqual(match, {u'a.py', u'aao.py', u'adir/'})

    def test_completion_more_args(self):
        event = MockEvent(u'%run a.py ')
        match = set(magic_run_completer(None, event))
        self.assertEqual(match, set(self.files + self.dirs))

    def test_completion_in_dir(self):
        event = MockEvent(u'%run a.py {}'.format(join(self.BASETESTDIR, 'a')))
        print(repr(event.line))
        match = set(magic_run_completer(None, event))
        self.assertEqual(match, {join(self.BASETESTDIR, f).replace('\\', '/') for f in (u'a.py', u'aao.py', u'aao.txt', u'adir/')})