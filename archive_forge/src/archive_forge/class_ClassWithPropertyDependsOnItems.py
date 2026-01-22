import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
class ClassWithPropertyDependsOnItems(ClassWithInstanceDefaultInit):
    """ Dummy class using depends_on to be compared with the one using
    observe."""
    discounted = Property(Bool(), depends_on='list_of_infos.age')
    discounted_n_calculations = Int()

    def _get_discounted(self):
        self.discounted_n_calculations += 1
        return any((info.age > 70 for info in self.list_of_infos))