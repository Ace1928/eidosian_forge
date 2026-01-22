import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
class TestPoentryPerPergraph(PoEntryTestCase):

    def test_single(self):
        self.exporter.poentry_per_paragraph('dummy', 10, 'foo\nbar\nbaz\n')
        self.check_output('                #: dummy:10\n                msgid ""\n                "foo\\n"\n                "bar\\n"\n                "baz\\n"\n                msgstr ""\n\n                ')

    def test_multi(self):
        self.exporter.poentry_per_paragraph('dummy', 10, 'spam\nham\negg\n\nSPAM\nHAM\nEGG\n')
        self.check_output('                #: dummy:10\n                msgid ""\n                "spam\\n"\n                "ham\\n"\n                "egg"\n                msgstr ""\n\n                #: dummy:14\n                msgid ""\n                "SPAM\\n"\n                "HAM\\n"\n                "EGG\\n"\n                msgstr ""\n\n                ')