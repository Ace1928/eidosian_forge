import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
class FooEnum(enum.Enum):
    foo = 0
    bar = 1
    baz = 2