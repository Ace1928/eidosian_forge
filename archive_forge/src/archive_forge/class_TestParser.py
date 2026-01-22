import unittest
from oslo_config import iniparser
class TestParser(iniparser.BaseParser):
    comment_called = False
    values = None
    section = ''

    def __init__(self):
        self.values = {}

    def assignment(self, key, value):
        self.values.setdefault(self.section, {})
        self.values[self.section][key] = value

    def new_section(self, section):
        self.section = section

    def comment(self, section):
        self.comment_called = True