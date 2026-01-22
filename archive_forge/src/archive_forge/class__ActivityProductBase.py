from collections import OrderedDict
import warnings
from ._util import get_backend
from .chemistry import Substance
from .units import allclose
class _ActivityProductBase(object):
    """Baseclass for activity products"""

    def __init__(self, stoich, *args):
        self.stoich = stoich
        self.args = args

    def __call__(self, c):
        pass