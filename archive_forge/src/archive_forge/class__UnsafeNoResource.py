from __future__ import annotations
import warnings
from typing import Sequence
from zope.interface import Attribute, Interface, implementer
from incremental import Version
from twisted.python.compat import nativeString
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.reflect import prefixedMethodNames
from twisted.web._responses import FORBIDDEN, NOT_FOUND
from twisted.web.error import UnsupportedMethod
class _UnsafeNoResource(_UnsafeErrorPage):
    """
    L{_UnsafeNoResource}, publicly available via the deprecated alias
    C{NoResource}, is a specialization of L{_UnsafeErrorPage} which
    returns the HTTP response code I{NOT FOUND}.

    Deprecated in Twisted 22.10.0 because it permits HTML injection; use
    L{twisted.web.pages.notFound} instead.
    """

    def __init__(self, message='Sorry. No luck finding that resource.'):
        _UnsafeErrorPage.__init__(self, NOT_FOUND, 'No Such Resource', message)