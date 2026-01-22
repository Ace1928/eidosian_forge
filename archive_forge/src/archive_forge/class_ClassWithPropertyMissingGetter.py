import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
class ClassWithPropertyMissingGetter(PersonInfo):
    """ Dummy class to test property that does not have getter/setter.
    """
    discounted = Property(Bool(), observe='age')