from __future__ import annotations
import operator
import types
import uuid
import warnings
from collections.abc import Sequence
from dataclasses import fields, is_dataclass, replace
from functools import partial
from tlz import concat, curry, merge, unique
from dask import config
from dask.base import (
from dask.base import tokenize as _tokenize
from dask.context import globalmethod
from dask.core import flatten, quote
from dask.highlevelgraph import HighLevelGraph
from dask.typing import Graph, NestedKeys
from dask.utils import (
def to_task_dask(expr):
    """Normalize a python object and merge all sub-graphs.

    - Replace ``Delayed`` with their keys
    - Convert literals to things the schedulers can handle
    - Extract dask graphs from all enclosed values

    Parameters
    ----------
    expr : object
        The object to be normalized. This function knows how to handle
        ``Delayed``s, as well as most builtin python types.

    Returns
    -------
    task : normalized task to be run
    dask : a merged dask graph that forms the dag for this task

    Examples
    --------
    >>> import dask
    >>> a = delayed(1, 'a')
    >>> b = delayed(2, 'b')
    >>> task, dask = to_task_dask([a, b, 3])  # doctest: +SKIP
    >>> task  # doctest: +SKIP
    ['a', 'b', 3]
    >>> dict(dask)  # doctest: +SKIP
    {'a': 1, 'b': 2}

    >>> task, dasks = to_task_dask({a: 1, b: 2})  # doctest: +SKIP
    >>> task  # doctest: +SKIP
    (dict, [['a', 1], ['b', 2]])
    >>> dict(dask)  # doctest: +SKIP
    {'a': 1, 'b': 2}
    """
    warnings.warn('The dask.delayed.to_dask_dask function has been Deprecated in favor of unpack_collections', stacklevel=2)
    if isinstance(expr, Delayed):
        return (expr.key, expr.dask)
    if is_dask_collection(expr):
        name = 'finalize-' + tokenize(expr, pure=True)
        keys = expr.__dask_keys__()
        opt = getattr(expr, '__dask_optimize__', dont_optimize)
        finalize, args = expr.__dask_postcompute__()
        dsk = {name: (finalize, keys) + args}
        dsk.update(opt(expr.__dask_graph__(), keys))
        return (name, dsk)
    if type(expr) is type(iter(list())):
        expr = list(expr)
    elif type(expr) is type(iter(tuple())):
        expr = tuple(expr)
    elif type(expr) is type(iter(set())):
        expr = set(expr)
    typ = type(expr)
    if typ in (list, tuple, set):
        args, dasks = unzip((to_task_dask(e) for e in expr), 2)
        args = list(args)
        dsk = merge(dasks)
        return (args, dsk) if typ is list else ((typ, args), dsk)
    if typ is dict:
        args, dsk = to_task_dask([[k, v] for k, v in expr.items()])
        return ((dict, args), dsk)
    if is_dataclass(expr):
        args, dsk = to_task_dask([[f.name, getattr(expr, f.name)] for f in fields(expr) if hasattr(expr, f.name)])
        return ((apply, typ, (), (dict, args)), dsk)
    if is_namedtuple_instance(expr):
        args, dsk = to_task_dask([v for v in expr])
        return ((typ, *args), dsk)
    if typ is slice:
        args, dsk = to_task_dask([expr.start, expr.stop, expr.step])
        return ((slice,) + tuple(args), dsk)
    return (expr, {})