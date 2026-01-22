import numpy
import cupy
from cupy import _core
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils
def asformat(self, format):
    """Return this matrix in a given sparse format.

        Args:
            format (str or None): Format you need.
        """
    if format is None or format == self.format:
        return self
    else:
        return getattr(self, 'to' + format)()