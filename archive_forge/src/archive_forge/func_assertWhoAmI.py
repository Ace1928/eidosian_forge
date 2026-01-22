from breezy import branch, config, errors, tests
from ..test_bedding import override_whoami
def assertWhoAmI(self, expected, *cmd_args, **kwargs):
    out, err = self.run_bzr(('whoami',) + cmd_args, **kwargs)
    self.assertEqual('', err)
    lines = out.splitlines()
    self.assertLength(1, lines)
    if isinstance(expected, bytes):
        expected = expected.decode(kwargs.get('encoding', 'ascii'))
    self.assertEqual(expected, lines[0].rstrip())