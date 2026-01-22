import itertools
from types import FunctionType
from zope.interface import Interface
from zope.interface import classImplements
from zope.interface.interface import InterfaceClass
from zope.interface.interface import _decorator_non_return
from zope.interface.interface import fromFunction
def __method_from_function(self, function, name):
    method = fromFunction(function, self, name=name)
    method.positional = method.positional[1:]
    return method