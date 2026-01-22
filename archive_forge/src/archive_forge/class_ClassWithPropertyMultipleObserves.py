import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
class ClassWithPropertyMultipleObserves(PersonInfo):
    """ Dummy class to test observing multiple values.
    """
    computed_value = Property(observe=[trait('age'), trait('gender')])
    computed_value_n_calculations = Int()

    def _get_computed_value(self):
        self.computed_value_n_calculations += 1
        return len(self.gender) + self.age