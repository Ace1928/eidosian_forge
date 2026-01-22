import unittest
from traits.api import (
from traits.observation.api import (
class SingleValue(HasTraits):
    value = Int()