import unittest
from traits.api import (
from traits.observation.api import (
class Potato(HasTraits):
    name = Str()