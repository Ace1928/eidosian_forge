import unittest
from traits.api import (
from traits.observation.api import (
class Crate(HasTraits):
    potato_bags = List(PotatoBag)