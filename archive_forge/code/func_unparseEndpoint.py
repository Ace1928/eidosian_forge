from typing import Mapping, Tuple
from zope.interface import implementer
from twisted.internet import interfaces
from twisted.internet.endpoints import (
from twisted.plugin import IPlugin
from . import proxyEndpoint
def unparseEndpoint(args: Tuple[object, ...], kwargs: Mapping[str, object]) -> str:
    """
    Un-parse the already-parsed args and kwargs back into endpoint syntax.

    @param args: C{:}-separated arguments

    @param kwargs: C{:} and then C{=}-separated keyword arguments

    @return: a string equivalent to the original format which this was parsed
        as.
    """
    description = ':'.join([quoteStringArgument(str(arg)) for arg in args] + sorted(('{}={}'.format(quoteStringArgument(str(key)), quoteStringArgument(str(value))) for key, value in kwargs.items())))
    return description