import itertools
import unittest.mock
from contextlib import contextmanager
from typing import Iterator
import torch
import torch._C
import torch._ops
import torch.utils._python_dispatch
import torch.utils._pytree as pytree
def all_py_loaded_overloads() -> Iterator[torch._ops.OpOverload]:
    """
    Warning: the set of overloads this will report is very subtle.  It is precisely
    the set of torch.ops functions that have actually been accessed from Python
    (e.g., we actually called torch.ops.aten.blah at some point.  This is DIFFERENT
    from the set of registered operators, which will in general be a larger set,
    as this would include all operators which we ran C++ static initializers or
    Python operator registration on.  This does not eagerly populate the list on
    torch.ops.aten; this list is lazy!

    In other words, this is good for traversing over everything that has an
    OpOverload object allocated in Python.  We use it for cache invalidation, but
    don't rely on this list being complete.

    Note that even if we did report all C++ registered overloads, this isn't guaranteed
    to be complete either, as a subsequent lazy load of a library which triggers more
    registrations could add more things to the set.
    """
    for ns in torch.ops:
        packets = getattr(torch.ops, ns)
        for op_name in packets:
            packet = getattr(packets, op_name)
            for overload in packet:
                yield getattr(packet, overload)