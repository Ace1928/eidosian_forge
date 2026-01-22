from .. import utils
from .._lazyload import h5py
from .._lazyload import tables
from decorator import decorator
def _is_tables(obj, allow_file=True, allow_group=True, allow_dataset=True):
    if not utils._try_import('tables'):
        return False
    else:
        types = []
        if allow_file:
            types.append(tables.File)
        if allow_group:
            types.append(tables.Group)
        if allow_dataset:
            types.append(tables.CArray)
            types.append(tables.Array)
        return isinstance(obj, tuple(types))