from collections.abc import Sequence
from typing import Optional, Union, cast
from twisted.python.compat import nativeString
from twisted.web._responses import RESPONSES
class MissingRenderMethod(RenderError):
    """
    Tried to use a render method which does not exist.

    @ivar element: The element which did not have the render method.
    @ivar renderName: The name of the renderer which could not be found.
    """

    def __init__(self, element, renderName):
        RenderError.__init__(self, element, renderName)
        self.element = element
        self.renderName = renderName

    def __repr__(self) -> str:
        return '{!r}: {!r} had no render method named {!r}'.format(self.__class__.__name__, self.element, self.renderName)