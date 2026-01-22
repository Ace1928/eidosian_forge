import unittest
from traits.api import CList, HasTraits, Instance, Int, List, Str, TraitError
class CFoo(HasTraits):
    ints = CList(Int)
    strs = CList(Str)