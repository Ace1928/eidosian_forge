import os.path as op
import inspect
import numpy as np
from ... import logging
from ..base import (
class DipyBaseInterface(LibraryBaseInterface):
    """A base interface for py:mod:`dipy` computations."""
    _pkg = 'dipy'