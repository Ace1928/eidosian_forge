from __future__ import annotations
from typing import Literal
from zope.interface import Interface, implementer
from twisted.python import components
@implementer(IX)
class XA(components.Adapter):

    def method(self) -> None:
        pass