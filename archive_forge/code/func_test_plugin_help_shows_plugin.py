import os
from ... import commands
from ..test_plugins import BaseTestPlugins
def test_plugin_help_shows_plugin(self):
    os.mkdir('plugin_test')
    source = "from breezy import commands\nclass cmd_myplug(commands.Command):\n    __doc__ = '''Just a simple test plugin.'''\n    aliases = ['mplg']\n    def run(self):\n        print ('Hello from my plugin')\n"
    self.create_plugin('myplug', source, 'plugin_test')
    self.load_with_paths(['plugin_test'])
    myplug = self.plugins['myplug'].module
    commands.register_command(myplug.cmd_myplug)
    self.addCleanup(commands.plugin_cmds.remove, 'myplug')
    help = self.run_bzr_utf8_out('help myplug')
    self.assertContainsRe(help, 'plugin "myplug"')
    help = self.split_help_commands()['myplug']
    self.assertContainsRe(help, '\\[myplug\\]')