import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
class OptionData:

    def __init__(self, name):
        self.name = name
        self.registry_keys = None
        self.error_messages = []

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name