import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
class GPGLLGPRMCFixQualities(Values):
    """
    The possible fix quality indications in GPGLL and GPRMC sentences.

    Unfortunately, these sentences only indicate whether data is good or void.
    They provide no other information, such as what went wrong if the data is
    void, or how good the data is if the data is not void.

    @cvar ACTIVE: The data is okay.
    @cvar VOID: The data is void, and should not be used.
    """
    ACTIVE = ValueConstant('A')
    VOID = ValueConstant('V')