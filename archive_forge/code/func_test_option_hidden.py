import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_option_hidden(self):
    opt = option.Option('hidden', help='Unseen.', hidden=True)
    self.assertEqual('', self.pot_from_option(opt))