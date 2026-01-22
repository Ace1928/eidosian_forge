import os
from inspect import isclass
from copy import deepcopy
from warnings import warn
from packaging.version import Version
from traits.trait_errors import TraitError
from traits.trait_handlers import TraitDictObject, TraitListObject
from ...utils.filemanip import md5, hash_infile, hash_timestamp
from .traits_extension import (
from ... import config, __version__
def _get_sorteddict(self, objekt, dictwithhash=False, hash_method=None, hash_files=True):
    if isinstance(objekt, dict):
        out = []
        for key, val in sorted(objekt.items()):
            if isdefined(val):
                out.append((key, self._get_sorteddict(val, dictwithhash, hash_method=hash_method, hash_files=hash_files)))
    elif isinstance(objekt, (list, tuple)):
        out = []
        for val in objekt:
            if isdefined(val):
                out.append(self._get_sorteddict(val, dictwithhash, hash_method=hash_method, hash_files=hash_files))
        if isinstance(objekt, tuple):
            out = tuple(out)
    else:
        out = None
        if isdefined(objekt):
            if hash_files and isinstance(objekt, (str, bytes)) and os.path.isfile(objekt):
                if hash_method is None:
                    hash_method = config.get('execution', 'hash_method')
                if hash_method.lower() == 'timestamp':
                    hash = hash_timestamp(objekt)
                elif hash_method.lower() == 'content':
                    hash = hash_infile(objekt)
                else:
                    raise Exception('Unknown hash method: %s' % hash_method)
                if dictwithhash:
                    out = (objekt, hash)
                else:
                    out = hash
            elif isinstance(objekt, float):
                out = _float_fmt(objekt)
            else:
                out = objekt
    return out