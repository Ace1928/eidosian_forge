import os
from breezy.errors import CommandError, NoSuchRevision
from breezy.tests import TestCaseWithTransport
from breezy.workingtree import WorkingTree
class TestRevisionInfo(TestCaseWithTransport):

    def check_output(self, output, *args):
        """Verify that the expected output matches what brz says.

        The output is supplied first, so that you can supply a variable
        number of arguments to bzr.
        """
        self.assertEqual(self.run_bzr(*args)[0], output)

    def test_revision_info(self):
        """Test that 'brz revision-info' reports the correct thing."""
        wt = self.make_branch_and_tree('.')
        wt.commit('Commit one', rev_id=b'a@r-0-1')
        wt.commit('Commit two', rev_id=b'a@r-0-1.1.1')
        wt.set_parent_ids([b'a@r-0-1', b'a@r-0-1.1.1'])
        wt.branch.set_last_revision_info(1, b'a@r-0-1')
        wt.commit('Commit three', rev_id=b'a@r-0-2')
        wt.controldir.destroy_workingtree()
        values = {'1': '1 a@r-0-1\n', '1.1.1': '1.1.1 a@r-0-1.1.1\n', '2': '2 a@r-0-2\n'}
        self.check_output(values['2'], 'revision-info')
        self.check_output(values['1'], 'revision-info 1')
        self.check_output(values['1.1.1'], 'revision-info 1.1.1')
        self.check_output(values['2'], 'revision-info 2')
        self.check_output(values['1'] + values['2'], 'revision-info 1 2')
        self.check_output('    ' + values['1'] + values['1.1.1'] + '    ' + values['2'], 'revision-info 1 1.1.1 2')
        self.check_output(values['2'] + values['1'], 'revision-info 2 1')
        self.check_output(values['1'], 'revision-info -r 1')
        self.check_output(values['1.1.1'], 'revision-info --revision 1.1.1')
        self.check_output(values['2'], 'revision-info -r 2')
        self.check_output(values['1'] + values['2'], 'revision-info -r 1..2')
        self.check_output('    ' + values['1'] + values['1.1.1'] + '    ' + values['2'], 'revision-info -r 1..1.1.1..2')
        self.check_output(values['2'] + values['1'], 'revision-info -r 2..1')
        self.check_output(values['1'], 'revision-info -r revid:a@r-0-1')
        self.check_output(values['1.1.1'], 'revision-info --revision revid:a@r-0-1.1.1')

    def test_revision_info_explicit_branch_dir(self):
        """Test that 'brz revision-info' honors the '-d' option."""
        wt = self.make_branch_and_tree('branch')
        wt.commit('Commit one', rev_id=b'a@r-0-1')
        self.check_output('1 a@r-0-1\n', 'revision-info -d branch')

    def test_revision_info_tree(self):
        wt = self.make_branch_and_tree('branch')
        wt.commit('Commit one', rev_id=b'a@r-0-1')
        wt.branch.create_checkout('checkout', lightweight=True)
        wt.commit('Commit two', rev_id=b'a@r-0-2')
        self.check_output('2 a@r-0-2\n', 'revision-info -d checkout')
        self.check_output('1 a@r-0-1\n', 'revision-info --tree -d checkout')

    def test_revision_info_tree_no_working_tree(self):
        b = self.make_branch('branch')
        out, err = self.run_bzr('revision-info --tree -d branch', retcode=3)
        self.assertEqual('', out)
        self.assertEqual('brz: ERROR: No WorkingTree exists for "branch".\n', err)

    def test_revision_info_not_in_history(self):
        builder = self.make_branch_builder('branch')
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None))], revision_id=b'A-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'B-id')
        builder.build_snapshot([b'A-id'], [], revision_id=b'C-id')
        builder.finish_series()
        self.check_output('  1 A-id\n??? B-id\n  2 C-id\n', 'revision-info -d branch revid:A-id revid:B-id revid:C-id')