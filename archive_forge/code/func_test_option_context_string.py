import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_option_context_string(self):
    s = 'Literally.'
    context = export_pot._ModuleContext('local.py', 3, ({}, {s: 17}))
    opt = option.Option('example', help=s)
    self.assertContainsString(self.pot_from_option(opt, context), "#: local.py:17\n# help of 'example' test\n")