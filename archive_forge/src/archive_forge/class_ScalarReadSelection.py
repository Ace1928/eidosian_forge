import numpy as np
from .. import h5s
class ScalarReadSelection:
    """
        Implements slicing for scalar datasets.
    """

    def __init__(self, fspace, args):
        if args == ():
            self.mshape = None
        elif args == (Ellipsis,):
            self.mshape = ()
        else:
            raise ValueError('Illegal slicing argument for scalar dataspace')
        self.mspace = h5s.create(h5s.SCALAR)
        self.fspace = fspace

    def __iter__(self):
        self.mspace.select_all()
        yield (self.fspace, self.mspace)