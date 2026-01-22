import os
import sys
from io import BytesIO
from types import CodeType, FunctionType
import dill
from packaging import version
from .. import config
@pklregister(set)
def _save_set(pickler, obj):
    log(pickler, f'Se: {obj}')
    try:
        args = (sorted(obj),)
    except Exception:
        from datasets.fingerprint import Hasher
        args = (sorted(obj, key=Hasher.hash),)
    pickler.save_reduce(set, args, obj=obj)
    log(pickler, '# Se')