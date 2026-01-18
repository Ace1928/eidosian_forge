import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
 Return a callable to be used with on_trait_change, which will
    inspect call signature.

    Parameters
    ----------
    mock_obj : mock.Mock
        Mock object for tracking calls.
    