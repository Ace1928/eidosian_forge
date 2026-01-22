from collections import OrderedDict
import numpy as _np
class OtherOptionEntity(object):
    """The parameter entity for general option, with a detailed value"""

    def __init__(self, val):
        self.val = val

    @classmethod
    def from_tvm(cls, x):
        """Build a OtherOptionEntity from autotvm.OtherOptionEntity

        Parameters
        ----------
        cls: class
            Calling class
        x: autotvm.OtherOptionEntity
            The source object

        Returns
        -------
        ret: OtherOptionEntity
            The corresponding OtherOptionEntity object
        """
        return cls(x.val)

    def __repr__(self):
        return str(self.val)