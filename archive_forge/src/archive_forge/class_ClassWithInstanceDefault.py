import unittest
from unittest import mock
from traits.api import (
from traits.observation import _has_traits_helpers as helpers
from traits.observation import expression
from traits.observation.observe import observe
class ClassWithInstanceDefault(HasTraits):
    instance_with_default = Instance(Bar, ())