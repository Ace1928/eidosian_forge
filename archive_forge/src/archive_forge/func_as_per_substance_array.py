import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def as_per_substance_array(self, cont, dtype='float64', unit=None, raise_on_unk=False):
    """Turns a dict into an ordered array

        Parameters
        ----------
        cont : array_like or dict
        dtype : str or numpy.dtype object
        unit : unit, optional
        raise_on_unk : bool

        """
    import numpy as np
    if isinstance(cont, np.ndarray):
        pass
    elif isinstance(cont, dict):
        substance_keys = self.substances.keys()
        if raise_on_unk:
            for k in cont:
                if k not in substance_keys:
                    raise KeyError('Unknown substance key: %s' % k)
        cont = [cont[k] for k in substance_keys]
    if unit is not None:
        cont = to_unitless(cont, unit)
    cont = np.atleast_1d(np.asarray(cont, dtype=dtype).squeeze())
    if cont.shape[-1] != self.ns:
        raise ValueError('Incorrect size')
    return cont * (unit if unit is not None else 1)