from typing import Dict
from zope.interface import implementer
from twisted.conch import avatar, error as econch, interfaces as iconch
from twisted.conch.insults import insults
from twisted.conch.ssh import factory, session
from twisted.python import components
class _Glue:
    """
    A feeble class for making one attribute look like another.

    This should be replaced with a real class at some point, probably.
    Try not to write new code that uses it.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, name):
        raise AttributeError(self.name, 'has no attribute', name)