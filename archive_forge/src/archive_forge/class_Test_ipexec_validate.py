import os
import unittest
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
class Test_ipexec_validate(tt.TempFileMixin):

    def test_main_path(self):
        """Test with only stdout results.
        """
        self.mktmp("print('A')\nprint('B')\n")
        out = 'A\nB'
        tt.ipexec_validate(self.fname, out)

    def test_main_path2(self):
        """Test with only stdout results, expecting windows line endings.
        """
        self.mktmp("print('A')\nprint('B')\n")
        out = 'A\r\nB'
        tt.ipexec_validate(self.fname, out)

    def test_exception_path(self):
        """Test exception path in exception_validate.
        """
        self.mktmp("import sys\nprint('A')\nprint('B')\nprint('C', file=sys.stderr)\nprint('D', file=sys.stderr)\n")
        out = 'A\nB'
        tt.ipexec_validate(self.fname, expected_out=out, expected_err='C\nD')

    def test_exception_path2(self):
        """Test exception path in exception_validate, expecting windows line endings.
        """
        self.mktmp("import sys\nprint('A')\nprint('B')\nprint('C', file=sys.stderr)\nprint('D', file=sys.stderr)\n")
        out = 'A\r\nB'
        tt.ipexec_validate(self.fname, expected_out=out, expected_err='C\r\nD')

    def tearDown(self):
        tt.TempFileMixin.tearDown(self)