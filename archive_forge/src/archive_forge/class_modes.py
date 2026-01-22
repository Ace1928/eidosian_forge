from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
class modes:
    """
    ECMA 48 standardized modes
    """
    KEYBOARD_ACTION = KAM = 2
    INSERTION_REPLACEMENT = IRM = 4
    LINEFEED_NEWLINE = LNM = 20