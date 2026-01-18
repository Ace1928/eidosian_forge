import inspect
import sys
from types import FunctionType
from types import MethodType
from zope.interface.exceptions import BrokenImplementation
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.exceptions import DoesNotImplement
from zope.interface.exceptions import Invalid
from zope.interface.exceptions import MultipleInvalid
from zope.interface.interface import Method
from zope.interface.interface import fromFunction
from zope.interface.interface import fromMethod
def verifyObject(iface, candidate, tentative=False):
    return _verify(iface, candidate, tentative, vtype='o')