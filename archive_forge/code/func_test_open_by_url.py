from breezy import branchbuilder, errors, tests, urlutils
from breezy.branch import Branch
from breezy.controldir import BranchReferenceLoop
from breezy.tests import per_controldir
from breezy.tests.features import UnicodeFilenameFeature
def test_open_by_url(self):
    if not self.bzrdir_format.is_supported():
        raise tests.TestNotApplicable('Control dir format not supported')
    t = self.get_transport()
    try:
        made_control = self.bzrdir_format.initialize(t.base)
    except errors.UninitializableFormat:
        raise tests.TestNotApplicable('Control dir does not support creating new branches.')
    made_control.create_repository()
    made_branch = self.create_branch(made_control, name='colo')
    other_branch = self.create_branch(made_control, name='othercolo')
    self.assertIsInstance(made_branch, Branch)
    self.assertEqual(made_control, made_branch.controldir)
    self.assertNotEqual(made_branch.user_url, other_branch.user_url)
    self.assertNotEqual(made_branch.control_url, other_branch.control_url)
    re_made_branch = Branch.open(made_branch.user_url)
    self.assertEqual(re_made_branch.name, 'colo')
    self.assertEqual(made_branch.control_url, re_made_branch.control_url)
    self.assertEqual(made_branch.user_url, re_made_branch.user_url)