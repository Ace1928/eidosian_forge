import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class DelegateTrait(HasTraits):
    value = Delegate('delegate')
    delegate = Trait(DelegatedFloatTrait())