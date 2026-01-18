import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_registry_option_title_context_string(self):
    s = 'Grounded!'
    context = export_pot._ModuleContext('practice.py', 3, ({}, {s: 144}))
    opt = option.RegistryOption.from_kwargs('concrete', title=s)
    self.assertContainsString(self.pot_from_option(opt, context), "#: practice.py:144\n# title of 'concrete' test\n")