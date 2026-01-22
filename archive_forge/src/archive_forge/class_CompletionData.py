import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
class CompletionData:

    def __init__(self):
        self.plugins = {}
        self.global_options = []
        self.commands = []

    def all_command_aliases(self):
        for c in self.commands:
            yield from c.aliases