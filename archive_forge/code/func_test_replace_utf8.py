from ...commands import Command, plugin_cmds, register_command
from .. import TestCaseWithMemoryTransport
def test_replace_utf8(self):

    def bzr(*args, **kwargs):
        kwargs['encoding'] = 'utf-8'
        return self.run_bzr_raw(*args, **kwargs)[0]
    register_command(cmd_echo_replace)
    try:
        self.assertEqual(b'foo', bzr('echo-replace foo'))
        self.assertEqual('fooµ'.encode(), bzr(['echo-replace', 'fooµ']))
    finally:
        plugin_cmds.remove('echo-replace')