import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_option_with_help(self):
    opt = option.Option('helpful', help='Info.')
    self.assertContainsString(self.pot_from_option(opt), '\n# help of \'helpful\' test\nmsgid "Info."\n')