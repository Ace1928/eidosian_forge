import collections
import functools
import numbers
import sys
from torch.utils.data.datapipes._hook_iterator import hook_iterator, _SnapshotState
from typing import (Any, Dict, Iterator, Generic, List, Set, Tuple, TypeVar, Union,
from typing import _eval_type, _tp_cache, _type_check, _type_repr  # type: ignore[attr-defined]
from typing import ForwardRef
from abc import ABCMeta
from typing import _GenericAlias  # type: ignore[attr-defined, no-redef]
def _dp_init_subclass(sub_cls, *args, **kwargs):
    sub_cls.reinforce_type = reinforce_type
    if getattr(sub_cls, '__type_class__', False):
        return
    if isinstance(sub_cls.type.param, ForwardRef):
        base_globals = sys.modules[sub_cls.__module__].__dict__
        try:
            param = _eval_type(sub_cls.type.param, base_globals, locals())
            sub_cls.type.param = param
        except TypeError as e:
            raise TypeError(f'{sub_cls.type.param.__forward_arg__} is not supported by Python typing') from e
    if '__iter__' in sub_cls.__dict__:
        iter_fn = sub_cls.__dict__['__iter__']
        hints = get_type_hints(iter_fn)
        if 'return' in hints:
            return_hint = hints['return']
            if return_hint == Iterator:
                return
            if not (hasattr(return_hint, '__origin__') and (return_hint.__origin__ == Iterator or return_hint.__origin__ == collections.abc.Iterator)):
                raise TypeError("Expected 'Iterator' as the return annotation for `__iter__` of {}, but found {}".format(sub_cls.__name__, _type_repr(hints['return'])))
            data_type = return_hint.__args__[0]
            if not issubtype(data_type, sub_cls.type.param):
                raise TypeError("Expected return type of '__iter__' as a subtype of {}, but found {} for {}".format(sub_cls.type, _type_repr(data_type), sub_cls.__name__))