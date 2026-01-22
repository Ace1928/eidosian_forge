import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
class cmd_zsh_completion(commands.Command):
    __doc__ = 'Generate a shell function for zsh command line completion.\n\n    This command generates a shell function which can be used by zsh to\n    automatically complete the currently typed command when the user presses\n    the completion key (usually tab).\n\n    Commonly used like this:\n        eval "`brz zsh-completion`"\n    '
    takes_options = [option.Option('function-name', short_name='f', type=str, argname='name', help='Name of the generated function (default: _brz)'), option.Option('debug', type=None, hidden=True, help='Enable shell code useful for debugging'), option.ListOption('plugin', type=str, argname='name', help='Enable completions for the selected plugin' + ' (default: all plugins)')]

    def run(self, **kwargs):
        if 'plugin' in kwargs:
            if len(kwargs['plugin']) > 0:
                kwargs['selected_plugins'] = kwargs['plugin']
            del kwargs['plugin']
        zsh_completion_function(sys.stdout, **kwargs)