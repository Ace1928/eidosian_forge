import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
class DelegateLeak(HasTraits):
    visible = Property(Bool, depends_on='can_enable')
    can_enable = DelegatesTo('flag', prefix='x')
    flag = Instance(Dummy, kw={'x': 42})