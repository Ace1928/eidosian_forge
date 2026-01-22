import functools as _functools
from collections import OrderedDict
from pickle import (
from struct import unpack
from sys import maxsize
from typing import Any, Dict, List
import torch
@_functools.lru_cache(maxsize=1)
def _get_allowed_globals():
    rc: Dict[str, Any] = {'collections.OrderedDict': OrderedDict, 'torch.nn.parameter.Parameter': torch.nn.Parameter, 'torch.serialization._get_layout': torch.serialization._get_layout, 'torch.Size': torch.Size, 'torch.Tensor': torch.Tensor}
    for t in [torch.complex32, torch.complex64, torch.complex128, torch.float8_e5m2, torch.float8_e4m3fn, torch.float16, torch.float32, torch.float64, torch.int8, torch.int16, torch.int32, torch.int64]:
        rc[str(t)] = t
    for tt in torch._tensor_classes:
        rc[f'{tt.__module__}.{tt.__name__}'] = tt
    for ts in torch._storage_classes:
        if ts not in (torch.storage.TypedStorage, torch.storage.UntypedStorage):
            rc[f'{ts.__module__}.{ts.__name__}'] = torch.serialization.StorageType(ts.__name__)
        else:
            rc[f'{ts.__module__}.{ts.__name__}'] = ts
    for f in [torch._utils._rebuild_parameter, torch._utils._rebuild_tensor, torch._utils._rebuild_tensor_v2, torch._utils._rebuild_tensor_v3, torch._utils._rebuild_sparse_tensor, torch._utils._rebuild_meta_tensor_no_storage, torch._utils._rebuild_nested_tensor]:
        rc[f'torch._utils.{f.__name__}'] = f
    rc['torch._tensor._rebuild_from_type_v2'] = torch._tensor._rebuild_from_type_v2
    return rc