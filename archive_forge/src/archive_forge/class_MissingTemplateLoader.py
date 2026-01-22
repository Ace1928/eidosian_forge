from collections.abc import Sequence
from typing import Optional, Union, cast
from twisted.python.compat import nativeString
from twisted.web._responses import RESPONSES
class MissingTemplateLoader(RenderError):
    """
    L{MissingTemplateLoader} is raised when trying to render an Element without
    a template loader, i.e. a C{loader} attribute.

    @ivar element: The Element which did not have a document factory.
    """

    def __init__(self, element):
        RenderError.__init__(self, element)
        self.element = element

    def __repr__(self) -> str:
        return f'{self.__class__.__name__!r}: {self.element!r} had no loader'