import unittest
from traits.api import CList, HasTraits, Instance, Int, List, Str, TraitError
class BazRef(HasTraits):
    bars = List(Bar, copy='ref')