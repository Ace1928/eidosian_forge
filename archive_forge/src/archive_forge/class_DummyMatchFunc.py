import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._testing import (
from traits.observation._trait_added_observer import (
from traits.trait_types import Str
class DummyMatchFunc:
    """ A callable to be used as TraitAddedObserver.match_func
    """

    def __init__(self, return_value):
        self.return_value = return_value

    def __call__(self, name, trait):
        return self.return_value

    def __eq__(self, other):
        return self.return_value == other.return_value

    def __hash__(self):
        return hash(self.return_value)