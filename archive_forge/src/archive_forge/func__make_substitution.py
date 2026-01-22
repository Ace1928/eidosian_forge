from abc import abstractmethod, ABCMeta
import collections
from collections import defaultdict
import collections.abc
import contextlib
import functools
import operator
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
import warnings
from types import WrapperDescriptorType, MethodWrapperType, MethodDescriptorType, GenericAlias
def _make_substitution(self, args, new_arg_by_param):
    """Create a list of new type arguments."""
    new_args = []
    for old_arg in args:
        if isinstance(old_arg, type):
            new_args.append(old_arg)
            continue
        substfunc = getattr(old_arg, '__typing_subst__', None)
        if substfunc:
            new_arg = substfunc(new_arg_by_param[old_arg])
        else:
            subparams = getattr(old_arg, '__parameters__', ())
            if not subparams:
                new_arg = old_arg
            else:
                subargs = []
                for x in subparams:
                    if isinstance(x, TypeVarTuple):
                        subargs.extend(new_arg_by_param[x])
                    else:
                        subargs.append(new_arg_by_param[x])
                new_arg = old_arg[tuple(subargs)]
        if self.__origin__ == collections.abc.Callable and isinstance(new_arg, tuple):
            new_args.extend(new_arg)
        elif _is_unpacked_typevartuple(old_arg):
            new_args.extend(new_arg)
        elif isinstance(old_arg, tuple):
            new_args.append(tuple(self._make_substitution(old_arg, new_arg_by_param)))
        else:
            new_args.append(new_arg)
    return new_args