from contextlib import suppress
import pickle
import numpy as np
from toolz import valmap, identity, partial
from .core import Interface
from .file import File
from .utils import frame, framesplit, suffix
class Numpy(Interface):

    def __init__(self, partd=None):
        if not partd or isinstance(partd, str):
            partd = File(partd)
        self.partd = partd
        Interface.__init__(self)

    def __getstate__(self):
        return {'partd': self.partd}

    def append(self, data, **kwargs):
        for k, v in data.items():
            self.partd.iset(suffix(k, '.dtype'), serialize_dtype(v.dtype))
        self.partd.append(valmap(serialize, data), **kwargs)

    def _get(self, keys, **kwargs):
        bytes = self.partd._get(keys, **kwargs)
        dtypes = self.partd._get([suffix(key, '.dtype') for key in keys], lock=False)
        dtypes = map(parse_dtype, dtypes)
        return list(map(deserialize, bytes, dtypes))

    def delete(self, keys, **kwargs):
        keys2 = [suffix(key, '.dtype') for key in keys]
        self.partd.delete(keys2, **kwargs)

    def _iset(self, key, value):
        return self.partd._iset(key, value)

    def drop(self):
        return self.partd.drop()

    def __del__(self):
        self.partd.__del__()

    @property
    def lock(self):
        return self.partd.lock

    def __exit__(self, *args):
        self.drop()
        self.partd.__exit__(self, *args)