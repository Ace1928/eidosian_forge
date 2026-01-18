import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_registry_option_title_context_missing(self):
    context = export_pot._ModuleContext('theory.py', 3)
    opt = option.RegistryOption.from_kwargs('abstract', title='Unfounded!')
    self.assertContainsString(self.pot_from_option(opt, context), "#: theory.py:3\n# title of 'abstract' test\n")