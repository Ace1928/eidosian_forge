from functools import lru_cache, wraps
from typing import List, Optional, Dict
from asgiref.sync import async_to_sync
from .mp_utils import multiproc, _MAX_PROCS, lazy_parallelize
def wrapper_multiproc(func):

    def wrapped_parallelize(*args, **kwargs):
        if dataset_callable:
            yield from lazy_parallelize(func, *args, processes=num_procs, result=dataset_callable(*args, **kwargs), **kwargs)
        else:
            datavar = dataset or kwargs.get(dataset_var)
            if datavar:
                yield from lazy_parallelize(func, *args, processes=num_procs, result=dataset, **kwargs)
        return func(*args, **kwargs)
    return wrapped_parallelize