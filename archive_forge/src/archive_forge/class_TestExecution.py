from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
class TestExecution(script.TestCaseWithTransportAndScript):

    def test_unknown_command(self):
        """A clear error is reported for commands that aren't recognised

        Testing the attributes of the SyntaxError instance is equivalent to
        using traceback.format_exception_only and comparing with:
          File "<string>", line 1
            foo --frob
            ^
        SyntaxError: Command not found "foo"
        """
        e = self.assertRaises(SyntaxError, self.run_script, '$ foo --frob')
        self.assertContainsRe(e.msg, 'not found.*foo')
        self.assertEqual(e.text, 'foo --frob')

    def test_blank_output_mismatches_output(self):
        """If you give output, the output must actually be blank.

        See <https://bugs.launchpad.net/bzr/+bug/637830>: previously blank
        output was a wildcard.  Now you must say ... if you want that.
        """
        self.assertRaises(AssertionError, self.run_script, '\n            $ echo foo\n            ')

    def test_null_output_matches_option(self):
        """If you want null output to be a wild card, you can pass
        null_output_matches_anything to run_script"""
        self.run_script('\n            $ echo foo\n            ', null_output_matches_anything=True)

    def test_ellipsis_everything(self):
        """A simple ellipsis matches everything."""
        self.run_script('\n        $ echo foo\n        ...\n        ')

    def test_ellipsis_matches_empty(self):
        self.run_script('\n        $ cd .\n        ...\n        ')

    def test_stops_on_unexpected_output(self):
        story = '\n$ mkdir dir\n$ cd dir\nThe cd command ouputs nothing\n'
        self.assertRaises(AssertionError, self.run_script, story)

    def test_stops_on_unexpected_error(self):
        story = '\n$ cat\n<Hello\n$ brz not-a-command\n'
        self.assertRaises(AssertionError, self.run_script, story)

    def test_continue_on_expected_error(self):
        story = '\n$ brz not-a-command\n2>..."not-a-command"\n'
        self.run_script(story)

    def test_continue_on_error_output(self):
        story = "\n$ brz init\n...\n$ cat >file\n<Hello\n$ brz add file\n...\n$ brz commit -m 'adding file'\n2>...\n"
        self.run_script(story)

    def test_ellipsis_output(self):
        story = '\n$ cat\n<first line\n<second line\n<last line\nfirst line\n...\nlast line\n'
        self.run_script(story)
        story = '\n$ brz not-a-command\n2>..."not-a-command"\n'
        self.run_script(story)
        story = '\n$ brz branch not-a-branch\n2>brz: ERROR: Not a branch...not-a-branch/".\n'
        self.run_script(story)