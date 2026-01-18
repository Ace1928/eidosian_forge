import os
from ... import commands
from ..test_plugins import BaseTestPlugins
def test_plugin_help_builtins_unaffected(self):
    help_commands = self.split_help_commands()
    for cmd_name in commands.builtin_command_names():
        if cmd_name in commands.plugin_command_names():
            continue
        try:
            help = commands.get_cmd_object(cmd_name).get_help_text()
        except NotImplementedError:
            pass
        else:
            self.assertNotContainsRe(help, 'plugin "[^"]*"')
        if cmd_name in help_commands:
            help = help_commands[cmd_name]
            self.assertNotContainsRe(help, 'plugin "[^"]*"')