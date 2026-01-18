from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_quoted_globbing(self):
    self.run_script("\n$ echo cat >cat\n$ cat '*'\n2>*: No such file or directory\n")