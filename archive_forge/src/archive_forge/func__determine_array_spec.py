from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict as ptDict, Type as ptType
import itertools
import weakref
from functools import cached_property
import numpy as np
from numba.core.utils import get_hashable_key
def _determine_array_spec(self, args):

    def validate_slice(s):
        return isinstance(s, slice) and s.start is None and (s.stop is None)
    if isinstance(args, (tuple, list)) and all(map(validate_slice, args)):
        ndim = len(args)
        if args[0].step == 1:
            layout = 'F'
        elif args[-1].step == 1:
            layout = 'C'
        else:
            layout = 'A'
    elif validate_slice(args):
        ndim = 1
        if args.step == 1:
            layout = 'C'
        else:
            layout = 'A'
    else:
        raise KeyError(f'Can only index numba types with slices with no start or stop, got {args}.')
    return (ndim, layout)