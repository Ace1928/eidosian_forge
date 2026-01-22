import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
class ClassWithPropertyTraitNotFound(HasTraits):
    """ Dummy class to test error, prevent issues like enthought/traits#447
    """
    person = Instance(PersonInfo)
    last_name = Property(observe='person.last_name')