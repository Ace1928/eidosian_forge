from breezy import controldir, errors, tests
from breezy.tests import per_controldir
class TestNoColocatedSupport(per_controldir.TestCaseWithControlDir):

    def make_controldir_with_repo(self):
        if not self.bzrdir_format.is_supported():
            raise tests.TestNotApplicable('Control dir format not supported')
        t = self.get_transport()
        try:
            made_control = self.make_controldir('.', format=self.bzrdir_format)
        except errors.UninitializableFormat:
            raise tests.TestNotApplicable('Control dir format not initializable')
        self.assertEqual(made_control._format, self.bzrdir_format)
        made_repo = made_control.create_repository()
        return made_control

    def test_destroy_colocated_branch(self):
        branch = self.make_branch('branch')
        self.assertRaises((controldir.NoColocatedBranchSupport, errors.UnsupportedOperation), branch.controldir.destroy_branch, 'colo')

    def test_create_colo_branch(self):
        made_control = self.make_controldir_with_repo()
        self.assertRaises(controldir.NoColocatedBranchSupport, made_control.create_branch, 'colo')

    def test_open_branch(self):
        made_control = self.make_controldir_with_repo()
        self.assertRaises(controldir.NoColocatedBranchSupport, made_control.open_branch, name='colo')

    def test_get_branch_reference(self):
        made_control = self.make_controldir_with_repo()
        self.assertRaises(controldir.NoColocatedBranchSupport, made_control.get_branch_reference, 'colo')

    def test_set_branch_reference(self):
        referenced = self.make_branch('referenced')
        made_control = self.make_controldir_with_repo()
        self.assertRaises(controldir.NoColocatedBranchSupport, made_control.set_branch_reference, referenced, name='colo')

    def test_get_branches(self):
        made_control = self.make_controldir_with_repo()
        made_control.create_branch()
        self.assertEqual(list(made_control.get_branches()), [''])