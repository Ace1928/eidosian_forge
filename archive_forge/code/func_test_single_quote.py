from Cython.Build.Dependencies import strip_string_literals
from Cython.TestUtils import CythonTest
def test_single_quote(self):
    self.t("'x'", "'_L1_'")