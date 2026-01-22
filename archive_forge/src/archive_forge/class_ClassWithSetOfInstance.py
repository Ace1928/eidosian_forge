import unittest
from traits.api import (
from traits.observation.api import (
class ClassWithSetOfInstance(HasTraits):
    instances = Set(Instance(SingleValue))
    instances_compat = Set(Instance(SingleValue))