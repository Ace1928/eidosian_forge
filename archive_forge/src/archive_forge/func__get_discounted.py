import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
def _get_discounted(self):
    self.discounted_n_calculations += 1
    return self.age > 60