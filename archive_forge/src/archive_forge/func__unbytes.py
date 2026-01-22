from __future__ import annotations
from .nbbase import nbformat, nbformat_minor
def _unbytes(obj):
    """There should be no bytes objects in a notebook

    v2 stores png/jpeg as b64 ascii bytes
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = _unbytes(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = _unbytes(v)
    elif isinstance(obj, bytes):
        obj = obj.decode('ascii')
    return obj