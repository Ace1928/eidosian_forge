import unittest
from traits.api import HasTraits, Instance, Str, Any, Property
class ClassWithClassAttribute(HasTraits):
    name = 'class defined name'
    foo = Str