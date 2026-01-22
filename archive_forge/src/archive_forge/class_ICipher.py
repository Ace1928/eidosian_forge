from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class ICipher(Interface):
    """
    A TLS cipher.
    """
    fullName = Attribute('The fully qualified name of the cipher in L{unicode}.')