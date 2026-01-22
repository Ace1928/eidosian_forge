import unittest
from traits.api import HasTraits, Instance, Str, Any, Property
class BazInstance(HasTraits):
    other = Instance('BarInstance')
    unique = Instance(Foo)
    shared = Instance(Foo)
    ref = Instance(Foo, copy='ref')