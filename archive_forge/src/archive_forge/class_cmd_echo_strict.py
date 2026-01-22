from ...commands import Command, plugin_cmds, register_command
from .. import TestCaseWithMemoryTransport
class cmd_echo_strict(cmd_echo_exact):
    """Raise a UnicodeError for unrepresentable characters."""
    encoding_type = 'strict'