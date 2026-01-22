from collections.abc import Sequence
import numpy as np
import xarray as xr
from numpy.linalg import LinAlgError
from scipy import special, stats
from . import _remove_indexes_to_reduce
from .linalg import cholesky, eigh
class XrRV:
    """Base random variable wrapper class.

    Most methods have a common signature between continuous and
    discrete variables in scipy. We define a base wrapper and
    then subclass it to add the specific methods like pdf or pmf.

    Notes
    -----
    One of the main goals of this library is ease of maintenance.
    We could wrap each distribution to preserve call signatures
    and avoid different behaviour between passing input arrays
    as args or kwargs, but so far we don't consider what we'd won
    doing this to be worth the extra maintenance burden.
    """

    def __init__(self, dist, *args, **kwargs):
        self.dist = dist
        self.args = args
        self.kwargs = kwargs

    def _broadcast_args(self, args, kwargs):
        """Broadcast and combine initialization and method provided args and kwargs."""
        len_args = len(args) + len(self.args)
        all_args = [*args, *self.args, *kwargs.values(), *self.kwargs.values()]
        broadcastable = []
        non_broadcastable = []
        b_idx = []
        n_idx = []
        for i, a in enumerate(all_args):
            if isinstance(a, xr.DataArray):
                broadcastable.append(a)
                b_idx.append(i)
            else:
                non_broadcastable.append(a)
                n_idx.append(i)
        broadcasted = list(xr.broadcast(*broadcastable))
        all_args = [x for x, _ in sorted(zip(broadcasted + non_broadcastable, b_idx + n_idx), key=lambda pair: pair[1])]
        all_keys = list(kwargs.keys()) + list(self.kwargs.keys())
        args = all_args[:len_args]
        kwargs = dict(zip(all_keys, all_args[len_args:]))
        return (args, kwargs)

    def rvs(self, *args, size=1, random_state=None, dims=None, apply_kwargs=None, **kwargs):
        """Implement base rvs method.

        In scipy, rvs has a common signature that doesn't depend on continuous
        or discrete, so we can define it here.
        """
        args, kwargs = self._broadcast_args(args, kwargs)
        size_in = tuple()
        dims_in = tuple()
        for a in (*args, *kwargs.values()):
            if isinstance(a, xr.DataArray):
                size_in = a.shape
                dims_in = a.dims
                break
        if isinstance(dims, str):
            dims = [dims]
        if isinstance(size, (Sequence, np.ndarray)):
            if dims is None:
                dims = [f'rv_dim{i}' for i, _ in enumerate(size)]
            if len(dims) != len(size):
                raise ValueError('dims and size must have the same length')
            size = (*size, *size_in)
        elif size > 1:
            if dims is None:
                dims = ['rv_dim0']
            if len(dims) != 1:
                raise ValueError('dims and size must have the same length')
            size = (size, *size_in)
        else:
            if size_in:
                size = size_in
            dims = None
        if dims is None:
            dims = tuple()
        if apply_kwargs is None:
            apply_kwargs = {}
        return xr.apply_ufunc(self.dist.rvs, *args, kwargs={**kwargs, 'size': size, 'random_state': random_state}, input_core_dims=[dims_in for _ in args], output_core_dims=[[*dims, *dims_in]], **apply_kwargs)