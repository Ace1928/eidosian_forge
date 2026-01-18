import re
import sys
import traceback
from collections import OrderedDict
from textwrap import dedent
from types import FunctionType
from typing import Callable, Dict, List, NoReturn, Optional, Tuple, cast
from xml.etree.ElementTree import XML
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.defer import (
from twisted.python.failure import Failure
from twisted.test.testutils import XMLAssertionMixin
from twisted.trial.unittest import SynchronousTestCase
from twisted.web._flatten import BUFFER_SIZE
from twisted.web.error import FlattenerError, UnfilledSlot, UnsupportedType
from twisted.web.iweb import IRenderable, IRequest, ITemplateLoader
from twisted.web.template import (
from twisted.web.test._util import FlattenTestCase
def test_serializedAttributeWithRenderable(self) -> None:
    """
        Like L{test_serializedAttributeWithTransparentTag}, but when the
        attribute is a provider of L{IRenderable} rather than a transparent
        tag.
        """

    @implementer(IRenderable)
    class Arbitrary:

        def __init__(self, value: Flattenable) -> None:
            self.value = value

        def render(self, request: Optional[IRequest]) -> Flattenable:
            return self.value

        def lookupRenderMethod(self, name: str) -> Callable[[Optional[IRequest], Tag], Flattenable]:
            raise NotImplementedError('Unexpected call')
    self.checkAttributeSanitization(Arbitrary, passthru)