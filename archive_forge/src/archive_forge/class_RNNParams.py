import warnings
import functools
from .. import symbol, init, ndarray
from ..base import string_types, numeric_types
class RNNParams(object):
    """Container for holding variables.
    Used by RNN cells for parameter sharing between cells.

    Parameters
    ----------
    prefix : str
        Names of all variables created by this container will
        be prepended with prefix.
    """

    def __init__(self, prefix=''):
        self._prefix = prefix
        self._params = {}

    def get(self, name, **kwargs):
        """Get the variable given a name if one exists or create a new one if missing.

        Parameters
        ----------
        name : str
            name of the variable
        **kwargs :
            more arguments that's passed to symbol.Variable
        """
        name = self._prefix + name
        if name not in self._params:
            self._params[name] = symbol.Variable(name, **kwargs)
        return self._params[name]