import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
class ClassWithPropertyDependsOnDefault(ClassWithInstanceDefaultInit):
    """ Dummy class for testing property with an observer on an extended
    attribute. 'info_with_default' has a default initializer.
    """
    extended_age = Property(depends_on='info_with_default.age')
    extended_age_n_calculations = Int()

    def _get_extended_age(self):
        self.extended_age_n_calculations += 1
        return self.info_with_default.age