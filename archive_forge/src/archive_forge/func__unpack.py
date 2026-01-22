from collections import OrderedDict
from collections.abc import Iterator
from operator import getitem
import uuid
import ray
from dask.base import quote
from dask.core import get as get_sync
from dask.utils import apply
def _unpack(expr):
    if isinstance(expr, ray.ObjectRef):
        token = expr.hex()
        repack_dsk[token] = (getitem, object_refs_token, len(object_refs))
        object_refs.append(expr)
        return token
    token = uuid.uuid4().hex
    typ = list if isinstance(expr, Iterator) else type(expr)
    if typ in (list, tuple, set):
        repack_task = (typ, [_unpack(i) for i in expr])
    elif typ in (dict, OrderedDict):
        repack_task = (typ, [[_unpack(k), _unpack(v)] for k, v in expr.items()])
    elif is_dataclass(expr):
        repack_task = (apply, typ, (), (dict, [[f.name, _unpack(getattr(expr, f.name))] for f in dataclass_fields(expr)]))
    else:
        return expr
    repack_dsk[token] = repack_task
    return token