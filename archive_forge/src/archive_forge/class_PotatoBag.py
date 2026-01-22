import unittest
from traits.api import (
from traits.observation.api import (
class PotatoBag(HasTraits):
    potatos = List(Instance(Potato))