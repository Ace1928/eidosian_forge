from __future__ import annotations
import re
import warnings
from io import BytesIO, StringIO
from incremental import Version, getVersionString
from twisted.python.compat import ioType
from twisted.python.util import InsensitiveDict
from twisted.web.sux import ParseError, XMLParser
class MismatchedTags(Exception):

    def __init__(self, filename, expect, got, endLine, endCol, begLine, begCol):
        self.filename, self.expect, self.got, self.begLine, self.begCol, self.endLine, self.endCol = (filename, expect, got, begLine, begCol, endLine, endCol)

    def __str__(self) -> str:
        return 'expected </%s>, got </%s> line: %s col: %s, began line: %s col: %s' % (self.expect, self.got, self.endLine, self.endCol, self.begLine, self.begCol)