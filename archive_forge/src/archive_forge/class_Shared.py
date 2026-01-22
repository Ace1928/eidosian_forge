import unittest
from traits.api import HasTraits, Instance, Str
class Shared(HasTraits):
    s = Str('new instance of Shared')