import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
def test_parse_fancy(self):
    ignored = ignores.parse_ignore_file(BytesIO(b'./rootdir\nrandomfile*\npath/from/ro?t\nunicode\xc2\xb5\ndos\r\n\n#comment\n xx \n!RE:^\\.z.*\n!!./.zcompdump\n'))
    self.assertEqual({'./rootdir', 'randomfile*', 'path/from/ro?t', 'unicodeµ', 'dos', ' xx ', '!RE:^\\.z.*', '!!./.zcompdump'}, ignored)