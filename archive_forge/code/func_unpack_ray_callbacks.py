import contextlib
from collections import namedtuple, defaultdict
from datetime import datetime
from dask.callbacks import Callback
def unpack_ray_callbacks(cbs):
    """Take an iterable of callbacks, return a list of each callback."""
    if cbs:
        return RayCallbacks(*([cb for cb in cbs_ if cb or CBS[idx] in CBS_DONT_DROP] or None for idx, cbs_ in enumerate(zip(*cbs))))
    else:
        return RayCallbacks(*[()] * len(CBS))