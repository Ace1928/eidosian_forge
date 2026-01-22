import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
class TestPoEntry(PoEntryTestCase):

    def test_simple(self):
        self.exporter.poentry('dummy', 1, 'spam')
        self.exporter.poentry('dummy', 2, 'ham', 'EGG')
        self.check_output('                #: dummy:1\n                msgid "spam"\n                msgstr ""\n\n                #: dummy:2\n                # EGG\n                msgid "ham"\n                msgstr ""\n\n                ')

    def test_duplicate(self):
        self.exporter.poentry('dummy', 1, 'spam')
        self.exporter.poentry('dummy', 2, 'spam', 'EGG')
        self.check_output('                #: dummy:1\n                msgid "spam"\n                msgstr ""\n\n                ')