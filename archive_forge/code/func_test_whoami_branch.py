from breezy import branch, config, errors, tests
from ..test_bedding import override_whoami
def test_whoami_branch(self):
    """branch specific user identity works."""
    wt = self.make_branch_and_tree('.')
    b = branch.Branch.open('.')
    self.set_branch_email(b, 'Branch Identity <branch@identi.ty>')
    self.assertWhoAmI('Branch Identity <branch@identi.ty>')
    self.assertWhoAmI('branch@identi.ty', '--email')
    self.overrideEnv('BRZ_EMAIL', 'Different ID <other@environ.ment>')
    self.assertWhoAmI('Different ID <other@environ.ment>')
    self.assertWhoAmI('other@environ.ment', '--email')