import unittest
from traits.api import BaseFloat, Either, Float, HasTraits, Str, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
class BadFloat(object):

    def __float__(self):
        raise ZeroDivisionError