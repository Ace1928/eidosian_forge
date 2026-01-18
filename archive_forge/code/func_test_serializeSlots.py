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
def test_serializeSlots(self) -> None:
    """
        Test that flattening a slot will use the slot value from the tag.
        """
    t1 = tags.p(slot('test'))
    t2 = t1.clone()
    t2.fillSlots(test='hello, world')
    self.assertFlatteningRaises(t1, UnfilledSlot)
    self.assertFlattensImmediately(t2, b'<p>hello, world</p>')