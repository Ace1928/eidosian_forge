import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_option_context_missing(self):
    context = export_pot._ModuleContext('remote.py', 3)
    opt = option.Option('metaphor', help='Not a literal in the source.')
    self.assertContainsString(self.pot_from_option(opt, context), "#: remote.py:3\n# help of 'metaphor' test\n")