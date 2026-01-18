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
def test_serializeCoroutineWithAwait(self) -> None:
    """
        Test that a coroutine returning an awaited deferred value is
        substituted with that value when flattened.
        """
    from textwrap import dedent
    namespace = dict(succeed=succeed)
    exec(dedent('\n            async def coro(x):\n                return await succeed(x)\n            '), namespace)
    coro = namespace['coro']
    self.assertFlattensImmediately(coro('four'), b'four')