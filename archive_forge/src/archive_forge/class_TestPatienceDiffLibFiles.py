import os
import shutil
import tempfile
import unittest
import patiencediff
from . import _patiencediff_py
class TestPatienceDiffLibFiles(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._PatienceSequenceMatcher = _patiencediff_py.PatienceSequenceMatcher_py
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.test_dir))

    def test_patience_unified_diff_files(self):
        txt_a = [b'hello there\n', b'world\n', b'how are you today?\n']
        txt_b = [b'hello there\n', b'how are you today?\n']
        with open(os.path.join(self.test_dir, 'a1'), 'wb') as f:
            f.writelines(txt_a)
        with open(os.path.join(self.test_dir, 'b1'), 'wb') as f:
            f.writelines(txt_b)
        unified_diff_files = patiencediff.unified_diff_files
        psm = self._PatienceSequenceMatcher
        old_pwd = os.getcwd()
        os.chdir(self.test_dir)
        try:
            self.assertEqual(['--- a1\n', '+++ b1\n', '@@ -1,3 +1,2 @@\n', ' hello there\n', '-world\n', ' how are you today?\n'], list(unified_diff_files('a1', 'b1', sequencematcher=psm)))
        finally:
            os.chdir(old_pwd)
        txt_a = [x + '\n' for x in 'abcdefghijklmnop']
        txt_b = [x + '\n' for x in 'abcdefxydefghijklmnop']
        with open(os.path.join(self.test_dir, 'a2'), 'w') as f:
            f.writelines(txt_a)
        with open(os.path.join(self.test_dir, 'b2'), 'w') as f:
            f.writelines(txt_b)
        os.chdir(self.test_dir)
        try:
            self.assertEqual(['--- a2\n', '+++ b2\n', '@@ -1,6 +1,11 @@\n', ' a\n', ' b\n', ' c\n', '+d\n', '+e\n', '+f\n', '+x\n', '+y\n', ' d\n', ' e\n', ' f\n'], list(unified_diff_files('a2', 'b2')))
            self.assertEqual(['--- a2\n', '+++ b2\n', '@@ -4,6 +4,11 @@\n', ' d\n', ' e\n', ' f\n', '+x\n', '+y\n', '+d\n', '+e\n', '+f\n', ' g\n', ' h\n', ' i\n'], list(unified_diff_files('a2', 'b2', sequencematcher=psm)))
        finally:
            os.chdir(old_pwd)