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
def test_observe_with_any_callables_accepting_one_argument(self):

    def handler_with_one_pos_arg(arg, *, optional=None):
        pass
    callables = [repr, lambda e: False, handler_with_one_pos_arg]
    for callable_ in callables:
        with self.subTest(callable=callable_):
            instance = ClassWithNumber()
            instance.observe(callable_, 'number')
            instance.number += 1