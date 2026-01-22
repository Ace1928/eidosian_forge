import numpy as np
from warnings import warn
from ._sputils import isintlike
def _raise_on_1d_array_slice(self):
    """We do not currently support 1D sparse arrays.

        This function is called each time that a 1D array would
        result, raising an error instead.

        Once 1D sparse arrays are implemented, it should be removed.
        """
    from scipy.sparse import sparray
    if isinstance(self, sparray):
        raise NotImplementedError('We have not yet implemented 1D sparse slices; please index using explicit indices, e.g. `x[:, [0]]`')