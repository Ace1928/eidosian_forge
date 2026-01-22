import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
class HasEnumInList(HasTraits):
    digits = Set(Int)
    digit_sequence = List(Enum(values='digits'))