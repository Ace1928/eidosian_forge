import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_registry_option_title(self):
    opt = option.RegistryOption.from_kwargs('group', help='Pick one.', title='Choose!')
    pot = self.pot_from_option(opt)
    self.assertContainsString(pot, '\n# title of \'group\' test\nmsgid "Choose!"\n')
    self.assertContainsString(pot, '\n# help of \'group\' test\nmsgid "Pick one."\n')