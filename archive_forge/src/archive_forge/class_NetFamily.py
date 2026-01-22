import binascii
import struct
from typing import Callable, Tuple, Type, Union
from zope.interface import implementer
from constantly import ValueConstant, Values
from typing_extensions import Literal
from twisted.internet import address
from twisted.python import compat
from . import _info, _interfaces
from ._exceptions import (
class NetFamily(Values):
    """
    Values for the 'family' field.
    """
    UNSPEC = ValueConstant(0)
    INET = ValueConstant(16)
    INET6 = ValueConstant(32)
    UNIX = ValueConstant(48)