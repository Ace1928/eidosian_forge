import re
import string
from zope.interface import implementer
from incremental import Version
from twisted.conch.insults import insults
from twisted.internet import defer, protocol, reactor
from twisted.logger import Logger
from twisted.python import _textattributes
from twisted.python.compat import iterbytes
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
def _scrollDown(self):
    self.y += 1
    if self.y >= self.height:
        self.y -= 1
        del self.lines[0]
        self.lines.append(self._emptyLine(self.width))