import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
def _with_callable_args(cls_or_self, **kwargs):
    """Wrapper that allows creation of class factories args that need to be
    called at construction time.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances and those arguments should only
    be calculated at construction time. Can be used in conjunction with _with_args

    Example::

        >>> # xdoctest: +SKIP("Undefined vars")
        >>> Foo.with_callable_args = classmethod(_with_callable_args)
        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_callable_args(cur_time=get_time_func).with_args(name="dan")
        >>> foo_instance1 = foo_builder()
        >>> # wait 50
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1.creation_time) == id(foo_instance2.creation_time)
        False
    """
    r = _PartialWrapper(partial(cls_or_self))
    return r.with_callable_args(**kwargs)