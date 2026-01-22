import sys
from typing import Optional, Sequence, Type
from zope.interface import Attribute, Interface
from twisted.plugin import getPlugins
from twisted.python import usage
class InvalidAuthArgumentString(StrcredException):
    """
    Raised by an authentication plugin when the argument string
    provided is formatted incorrectly.
    """