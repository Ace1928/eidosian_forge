from ...commands import Command, plugin_cmds, register_command
from .. import TestCaseWithMemoryTransport
def test_strict_utf8(self):

    def bzr(*args, **kwargs):
        kwargs['encoding'] = 'utf-8'
        return self.run_bzr_raw(*args, **kwargs)[0]
    register_command(cmd_echo_strict)
    try:
        self.assertEqual(b'foo', bzr('echo-strict foo'))
        expected = 'fooµ'
        expected = expected.encode('utf-8')
        self.assertEqual(expected, bzr(['echo-strict', 'fooµ']))
    finally:
        plugin_cmds.remove('echo-strict')