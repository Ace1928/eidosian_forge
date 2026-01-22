import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def _init_trait_listeners(self):
    """ Force the DelegatesTo listener to hook up first to exercise the
        worst case.
        """
    for name in ['y', '_on_dummy1_x']:
        data = self.__class__.__listener_traits__[name]
        getattr(self, '_init_trait_%s_listener' % data[0])(name, *data)