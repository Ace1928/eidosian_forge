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
def test_observe_different_target(self):
    parent1 = ClassWithInstance()
    parent2 = ClassWithInstance()
    instance = ClassWithNumber()
    parent1.instance = instance
    parent2.instance = instance
    handler = mock.Mock()
    observe(object=parent1, expression=trait('instance').trait('number'), handler=handler)
    observe(object=parent2, expression=trait('instance').trait('number'), handler=handler)
    instance.number += 1
    self.assertEqual(handler.call_count, 2)