import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
class OtherEnum(enum.Enum):
    one = 1
    two = 2
    three = 3