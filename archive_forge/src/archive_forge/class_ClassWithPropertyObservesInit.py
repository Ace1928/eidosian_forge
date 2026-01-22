import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
class ClassWithPropertyObservesInit(ClassWithInstanceDefaultInit):
    """ Dummy class for testing property with an observer on an extended
    attribute. sample_info has a default value depending on
    'info_without_default'. The value of 'info_without_default' is provided
    by __init__.
    To be compared with the next class using depends_on.
    """
    extended_age = Property(observe='sample_info.age')
    extended_age_n_calculations = Int()

    def _get_extended_age(self):
        self.extended_age_n_calculations += 1
        return self.sample_info.age