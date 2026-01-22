import argparse
import functools
from cliff import command
from cliff.tests import base
class TestDescription(base.TestBase):

    def test_get_description_docstring(self):
        cmd = TestCommand(None, None)
        desc = cmd.get_description()
        assert desc == 'Description of command.\n    '

    def test_get_description_attribute(self):
        cmd = TestCommand(None, None)
        cmd._description = 'this is not the default'
        desc = cmd.get_description()
        assert desc == 'this is not the default'

    def test_get_description_default(self):
        cmd = TestCommandNoDocstring(None, None)
        desc = cmd.get_description()
        assert desc == ''