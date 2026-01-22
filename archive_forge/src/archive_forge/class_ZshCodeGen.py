import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
class ZshCodeGen:
    """Generate a zsh script for given completion data."""

    def __init__(self, data, function_name='_brz', debug=False):
        self.data = data
        self.function_name = function_name
        self.debug = debug

    def script(self):
        return '#compdef brz bzr\n\n%(function_name)s ()\n{\n    local ret=1\n    local -a args\n    args+=(\n%(global-options)s\n    )\n\n    _arguments $args[@] && ret=0\n\n    return ret\n}\n\n%(function_name)s\n' % {'global-options': self.global_options(), 'function_name': self.function_name}

    def global_options(self):
        lines = []
        for long, short, help in self.data.global_options:
            lines.append("      '({}{}){}[{}]'".format(short + ' ' if short else '', long, long, help))
        return '\n'.join(lines)