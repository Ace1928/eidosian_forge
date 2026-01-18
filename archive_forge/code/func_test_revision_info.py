import os
from breezy.errors import CommandError, NoSuchRevision
from breezy.tests import TestCaseWithTransport
from breezy.workingtree import WorkingTree
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