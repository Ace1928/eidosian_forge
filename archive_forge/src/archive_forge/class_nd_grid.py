import math
import numpy
import cupy
from cupy import _core
class nd_grid(object):
    """Construct a multi-dimensional "meshgrid".

    ``grid = nd_grid()`` creates an instance which will return a mesh-grid
    when indexed.  The dimension and number of the output arrays are equal
    to the number of indexing dimensions.  If the step length is not a
    complex number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then the
    integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    If instantiated with an argument of ``sparse=True``, the mesh-grid is
    open (or not fleshed out) so that only one-dimension of each returned
    argument is greater than 1.

    Args:
        sparse (bool, optional): Whether the grid is sparse or not.
            Default is False.

    .. seealso:: :data:`numpy.mgrid` and :data:`numpy.ogrid`

    """

    def __init__(self, sparse=False):
        self.sparse = sparse

    def __getitem__(self, key):
        if isinstance(key, slice):
            step = key.step
            stop = key.stop
            start = key.start
            if start is None:
                start = 0
            if isinstance(step, complex):
                step = abs(step)
                length = int(step)
                if step != 1:
                    step = (key.stop - start) / float(step - 1)
                stop = key.stop + step
                return cupy.arange(0, length, 1, float) * step + start
            else:
                return cupy.arange(start, stop, step)
        size = []
        typ = int
        for k in range(len(key)):
            step = key[k].step
            start = key[k].start
            if start is None:
                start = 0
            if step is None:
                step = 1
            if isinstance(step, complex):
                size.append(int(abs(step)))
                typ = float
            else:
                size.append(int(math.ceil((key[k].stop - start) / (step * 1.0))))
            if isinstance(step, float) or isinstance(start, float) or isinstance(key[k].stop, float):
                typ = float
        if self.sparse:
            nn = [cupy.arange(_x, dtype=_t) for _x, _t in zip(size, (typ,) * len(size))]
        else:
            nn = cupy.indices(size, typ)
        for k in range(len(size)):
            step = key[k].step
            start = key[k].start
            if start is None:
                start = 0
            if step is None:
                step = 1
            if isinstance(step, complex):
                step = int(abs(step))
                if step != 1:
                    step = (key[k].stop - start) / float(step - 1)
            nn[k] = nn[k] * step + start
        if self.sparse:
            slobj = [cupy.newaxis] * len(size)
            for k in range(len(size)):
                slobj[k] = slice(None, None)
                nn[k] = nn[k][tuple(slobj)]
                slobj[k] = cupy.newaxis
        return nn

    def __len__(self):
        return 0