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
def _incompat(required, implemented):
    if len(implemented['required']) > len(required['required']):
        return _MSG_TOO_MANY
    if len(implemented['positional']) < len(required['positional']) and (not implemented['varargs']):
        return "implementation doesn't allow enough arguments"
    if required['kwargs'] and (not implemented['kwargs']):
        return "implementation doesn't support keyword arguments"
    if required['varargs'] and (not implemented['varargs']):
        return "implementation doesn't support variable arguments"