from breezy import conflicts, tests
from breezy.bzr import conflicts as _mod_bzr_conflicts
from breezy.tests import KnownFailure, script
from breezy.tests.blackbox import test_conflicts
class TestResolveAuto(tests.TestCaseWithTransport):

    def test_auto_resolve(self):
        """Text conflicts can be resolved automatically"""
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/file', b'<<<<<<<\na\n=======\n>>>>>>>\n')])
        tree.add('file', ids=b'file_id')
        self.assertEqual(tree.kind('file'), 'file')
        file_conflict = _mod_bzr_conflicts.TextConflict('file', file_id=b'file_id')
        tree.set_conflicts([file_conflict])
        note = self.run_bzr('resolve', retcode=1, working_dir='tree')[1]
        self.assertContainsRe(note, '0 conflicts auto-resolved.')
        self.assertContainsRe(note, 'Remaining conflicts:\nText conflict in file')
        self.build_tree_contents([('tree/file', b'a\n')])
        note = self.run_bzr('resolve', working_dir='tree')[1]
        self.assertContainsRe(note, 'All conflicts resolved.')