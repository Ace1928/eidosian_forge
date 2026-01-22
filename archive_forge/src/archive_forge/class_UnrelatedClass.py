import unittest
from traits.api import HasTraits, Subclass, TraitError, Type
class UnrelatedClass:
    """ Does not extend the BaseClass """
    pass