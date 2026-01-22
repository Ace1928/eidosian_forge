import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.trait_base import Undefined, Uninitialized
from traits.trait_types import Float, Instance, Int, List
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._testing import (
class DummyParent(HasTraits):
    number = Int()
    number2 = Int()
    instance = Instance(Dummy, allow_none=True)
    instance2 = Instance(Dummy)
    income = Float()
    dummies = List(Dummy)