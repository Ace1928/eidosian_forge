from ...commands import Command, plugin_cmds, register_command
from .. import TestCaseWithMemoryTransport
class cmd_echo_replace(cmd_echo_exact):
    """Replace bogus unicode characters."""
    encoding_type = 'replace'