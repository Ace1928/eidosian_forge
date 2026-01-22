import unittest
from traits.api import CList, HasTraits, Instance, Int, List, Str, TraitError
class DeepBazBazRef(HasTraits):
    baz = Instance(BazRef)