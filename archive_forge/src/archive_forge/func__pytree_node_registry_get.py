from __future__ import annotations
import dataclasses
import functools
import inspect
import sys
from collections import OrderedDict, defaultdict, deque, namedtuple
from operator import methodcaller
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple, Sequence, overload
from typing_extensions import Self  # Python 3.11+
from optree import _C
from optree.typing import (
from optree.utils import safe_zip, total_order_sorted, unzip2
def _pytree_node_registry_get(cls: type, *, namespace: str=__GLOBAL_NAMESPACE) -> PyTreeNodeRegistryEntry | None:
    handler: PyTreeNodeRegistryEntry | None = _NODETYPE_REGISTRY.get(cls)
    if handler is not None:
        return handler
    handler = _NODETYPE_REGISTRY.get((namespace, cls))
    if handler is not None:
        return handler
    if is_structseq_class(cls):
        return _NODETYPE_REGISTRY.get(structseq)
    if is_namedtuple_class(cls):
        return _NODETYPE_REGISTRY.get(namedtuple)
    return None