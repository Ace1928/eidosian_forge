import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_notifiers_on_trait(self):

    class Foo(HasTraits):
        x = Int()

        def _x_changed(self):
            pass
    foo = Foo(x=1)
    x_ctrait = foo.trait('x')
    tnotifiers = x_ctrait._notifiers(True)
    self.assertEqual(len(tnotifiers), 1)
    notifier, = tnotifiers
    self.assertEqual(notifier.handler, Foo._x_changed)