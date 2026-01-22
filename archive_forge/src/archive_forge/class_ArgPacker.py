from collections import deque
from numba.core import types, cgutils
class ArgPacker(object):
    """
    Compute the position for each high-level typed argument.
    It flattens every composite argument into primitive types.
    It maintains a position map for unflattening the arguments.

    Since struct (esp. nested struct) have specific ABI requirements (e.g.
    alignment, pointer address-space, ...) in different architecture (e.g.
    OpenCL, CUDA), flattening composite argument types simplifes the call
    setup from the Python side.  Functions are receiving simple primitive
    types and there are only a handful of these.
    """

    def __init__(self, dmm, fe_args):
        self._dmm = dmm
        self._fe_args = fe_args
        self._nargs = len(fe_args)
        self._dm_args = []
        argtys = []
        for ty in fe_args:
            dm = self._dmm.lookup(ty)
            self._dm_args.append(dm)
            argtys.append(dm.get_argument_type())
        self._unflattener = _Unflattener(argtys)
        self._be_args = list(_flatten(argtys))

    def as_arguments(self, builder, values):
        """Flatten all argument values
        """
        if len(values) != self._nargs:
            raise TypeError('invalid number of args: expected %d, got %d' % (self._nargs, len(values)))
        if not values:
            return ()
        args = [dm.as_argument(builder, val) for dm, val in zip(self._dm_args, values)]
        args = tuple(_flatten(args))
        return args

    def from_arguments(self, builder, args):
        """Unflatten all argument values
        """
        valtree = self._unflattener.unflatten(args)
        values = [dm.from_argument(builder, val) for dm, val in zip(self._dm_args, valtree)]
        return values

    def assign_names(self, args, names):
        """Assign names for each flattened argument values.
        """
        valtree = self._unflattener.unflatten(args)
        for aval, aname in zip(valtree, names):
            self._assign_names(aval, aname)

    def _assign_names(self, val_or_nested, name, depth=()):
        if isinstance(val_or_nested, (tuple, list)):
            for pos, aval in enumerate(val_or_nested):
                self._assign_names(aval, name, depth=depth + (pos,))
        else:
            postfix = '.'.join(map(str, depth))
            parts = [name, postfix]
            val_or_nested.name = '.'.join(filter(bool, parts))

    @property
    def argument_types(self):
        """Return a list of LLVM types that are results of flattening
        composite types.
        """
        return tuple((ty for ty in self._be_args if ty != ()))