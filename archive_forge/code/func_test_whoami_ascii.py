from breezy import branch, config, errors, tests
from ..test_bedding import override_whoami
def test_whoami_ascii(self):
    """
        verify that whoami doesn't totally break when in utf-8, using an ascii
        encoding.
        """
    wt = self.make_branch_and_tree('.')
    b = branch.Branch.open('.')
    self.set_branch_email(b, 'Branch Identity € <branch@identi.ty>')
    self.assertWhoAmI('Branch Identity ? <branch@identi.ty>', encoding='ascii')
    self.assertWhoAmI('branch@identi.ty', '--email', encoding='ascii')