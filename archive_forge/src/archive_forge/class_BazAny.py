import unittest
from traits.api import HasTraits, Instance, Str, Any, Property
class BazAny(HasTraits):
    other = Any