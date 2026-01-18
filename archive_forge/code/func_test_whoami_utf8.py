from breezy import branch, config, errors, tests
from ..test_bedding import override_whoami
def test_whoami_utf8(self):
    """verify that an identity can be in utf-8."""
    self.run_bzr(['whoami', 'Branch Identity € <branch@identi.ty>'], encoding='utf-8')
    self.assertWhoAmI(b'Branch Identity \xe2\x82\xac <branch@identi.ty>', encoding='utf-8')
    self.assertWhoAmI('branch@identi.ty', '--email')