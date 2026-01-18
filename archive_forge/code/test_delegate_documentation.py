import unittest
from traits.api import Delegate, HasTraits, Instance, Str
 Test that a delegated trait may be reset.

        Deleting the attribute should reset the trait back to its initial
        delegation behavior.
        