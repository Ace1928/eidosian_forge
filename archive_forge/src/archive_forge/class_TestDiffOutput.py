import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
class TestDiffOutput(DiffBase):

    def test_diff_output(self):
        self.make_example_branch()
        self.build_tree_contents([('hello', b'hello world!\n')])
        output = self.run_brz_subprocess('diff', retcode=1)[0]
        self.assertTrue(b'\n+hello world!\n' in output)