import unittest
from traits.api import (
from traits.observation.api import (
class ClassWithListOfInstance(HasTraits):
    list_of_instances = List(Instance(SingleValue))