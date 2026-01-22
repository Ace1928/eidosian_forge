import numpy as np
from .base import product
from .. import h5s, h5r, _selector
class SimpleSelection(Selection):
    """ A single "rectangular" (regular) selection composed of only slices
        and integer arguments.  Can participate in broadcasting.
    """

    @property
    def mshape(self):
        """ Shape of current selection """
        return self._sel[1]

    @property
    def array_shape(self):
        scalar = self._sel[3]
        return tuple((x for x, s in zip(self.mshape, scalar) if not s))

    def __init__(self, shape, spaceid=None, hyperslab=None):
        super().__init__(shape, spaceid)
        if hyperslab is not None:
            self._sel = hyperslab
        else:
            rank = len(self.shape)
            self._sel = ((0,) * rank, self.shape, (1,) * rank, (False,) * rank)

    def expand_shape(self, source_shape):
        """Match the dimensions of an array to be broadcast to the selection

        The returned shape describes an array of the same size as the input
        shape, but its dimensions

        E.g. with a dataset shape (10, 5, 4, 2), writing like this::

            ds[..., 0] = np.ones((5, 4))

        The source shape (5, 4) will expand to (1, 5, 4, 1).
        Then the broadcast method below repeats that chunk 10
        times to write to an effective shape of (10, 5, 4, 1).
        """
        start, count, step, scalar = self._sel
        rank = len(count)
        remaining_src_dims = list(source_shape)
        eshape = []
        for idx in range(1, rank + 1):
            if len(remaining_src_dims) == 0 or scalar[-idx]:
                eshape.append(1)
            else:
                t = remaining_src_dims.pop()
                if t == 1 or count[-idx] == t:
                    eshape.append(t)
                else:
                    raise TypeError("Can't broadcast %s -> %s" % (source_shape, self.array_shape))
        if any([n > 1 for n in remaining_src_dims]):
            raise TypeError("Can't broadcast %s -> %s" % (source_shape, self.array_shape))
        return tuple(eshape[::-1])

    def broadcast(self, source_shape):
        """ Return an iterator over target dataspaces for broadcasting.

        Follows the standard NumPy broadcasting rules against the current
        selection shape (self.mshape).
        """
        if self.shape == ():
            if product(source_shape) != 1:
                raise TypeError("Can't broadcast %s to scalar" % source_shape)
            self._id.select_all()
            yield self._id
            return
        start, count, step, scalar = self._sel
        rank = len(count)
        tshape = self.expand_shape(source_shape)
        chunks = tuple((x // y for x, y in zip(count, tshape)))
        nchunks = product(chunks)
        if nchunks == 1:
            yield self._id
        else:
            sid = self._id.copy()
            sid.select_hyperslab((0,) * rank, tshape, step)
            for idx in range(nchunks):
                offset = tuple((x * y * z + s for x, y, z, s in zip(np.unravel_index(idx, chunks), tshape, step, start)))
                sid.offset_simple(offset)
                yield sid