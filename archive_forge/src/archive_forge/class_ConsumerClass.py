import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class ConsumerClass(HasTraits):
    x = Instance(FactoryClass, ())