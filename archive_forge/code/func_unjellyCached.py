import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
def unjellyCached(unjellier, unjellyList):
    luid = unjellyList[1]
    return unjellier.invoker.cachedLocallyAs(luid)._borgify()