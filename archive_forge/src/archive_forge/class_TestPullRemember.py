from typing import List
from breezy import branch, urlutils
from breezy.tests import script
class TestPullRemember(script.TestCaseWithTransportAndScript, TestRememberMixin):
    working_dir = 'work'
    command = ['pull']
    first_use_args = ['../parent']
    next_uses_args = ['../new_parent']

    def setUp(self):
        super().setUp()
        self.run_script("\n            $ brz init parent\n            $ cd parent\n            $ echo parent > file\n            $ brz add\n            $ brz commit -m 'initial commit'\n            $ cd ..\n            $ brz init {working_dir}\n            ".format(working_dir=self.working_dir), null_output_matches_anything=True)

    def setup_next_uses(self):
        self.do_command(*self.first_use_args)
        self.run_script("\n            $ brz branch parent new_parent\n            $ cd new_parent\n            $ echo new parent > file\n            $ brz commit -m 'new parent'\n            $ cd ..\n            ", null_output_matches_anything=True)

    def assertLocations(self, expected_locations):
        br, _ = branch.Branch.open_containing(self.working_dir)
        if not expected_locations:
            self.assertEqual(None, br.get_parent())
        else:
            expected_pull_location = expected_locations[0]
            pull_location = urlutils.relative_url(br.base, br.get_parent())
            self.assertIsSameRealPath(expected_pull_location, pull_location)