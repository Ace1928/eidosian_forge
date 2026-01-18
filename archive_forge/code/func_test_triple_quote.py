from Cython.Build.Dependencies import strip_string_literals
from Cython.TestUtils import CythonTest
def test_triple_quote(self):
    self.t(" '''a\n''' ", " '''_L1_''' ")