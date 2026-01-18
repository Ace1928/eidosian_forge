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
def toVT102(self):
    attrs = []
    if self._subtracting:
        attrs.append(0)
    if self.bold:
        attrs.append(insults.BOLD)
    if self.underline:
        attrs.append(insults.UNDERLINE)
    if self.blink:
        attrs.append(insults.BLINK)
    if self.reverseVideo:
        attrs.append(insults.REVERSE_VIDEO)
    if self.foreground != WHITE:
        attrs.append(FOREGROUND + self.foreground)
    if self.background != BLACK:
        attrs.append(BACKGROUND + self.background)
    if attrs:
        return '\x1b[' + ';'.join(map(str, attrs)) + 'm'
    return ''