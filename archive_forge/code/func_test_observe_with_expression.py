import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.trait_types import Instance, Int
from traits.observation.api import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation.expression import compile_expr, trait
from traits.observation.observe import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_observe_with_expression(self):
    foo = ClassWithNumber()
    handler = mock.Mock()
    observe(object=foo, expression=trait('number'), handler=handler)
    foo.number += 1
    self.assertEqual(handler.call_count, 1)
    handler.reset_mock()
    observe(object=foo, expression=trait('number'), handler=handler, remove=True)
    foo.number += 1
    self.assertEqual(handler.call_count, 0)