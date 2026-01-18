import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner
def test_unshelve_keep(self):
    tree = self.make_branch_and_tree('.')
    tree.commit('make root')
    self.build_tree(['file'])
    sr = ScriptRunner()
    sr.run_script(self, '\n$ brz add file\nadding file\n$ brz shelve --all -m Foo\n2>Selected changes:\n2>-D  file\n2>Changes shelved with id "1".\n$ brz shelve --list\n  1: Foo\n$ brz unshelve --keep\n2>Using changes with id "1".\n2>Message: Foo\n2>+N  file\n2>All changes applied successfully.\n$ brz shelve --list\n  1: Foo\n$ cat file\ncontents of file\n')