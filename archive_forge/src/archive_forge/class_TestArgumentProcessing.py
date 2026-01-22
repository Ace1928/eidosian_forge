from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
class TestArgumentProcessing(script.TestCaseWithTransportAndScript):

    def test_globing(self):
        self.run_script('\n$ echo cat >cat\n$ echo dog >dog\n$ cat *\ncat\ndog\n')

    def test_quoted_globbing(self):
        self.run_script("\n$ echo cat >cat\n$ cat '*'\n2>*: No such file or directory\n")

    def test_quotes_removal(self):
        self.run_script('\n$ echo \'cat\' "dog" \'"chicken"\' "\'dragon\'"\ncat dog "chicken" \'dragon\'\n')

    def test_verbosity_isolated(self):
        """Global verbosity is isolated from commands run in scripts.
        """
        self.run_script('\n        $ brz init --quiet a\n        ')
        self.assertEqual(trace.is_quiet(), False)