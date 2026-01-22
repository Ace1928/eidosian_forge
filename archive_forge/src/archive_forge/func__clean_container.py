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
def _clean_container(self, objekt, undefinedval=None, skipundefined=False):
    """Convert a traited obejct into a pure python representation."""
    if isinstance(objekt, TraitDictObject) or isinstance(objekt, dict):
        out = {}
        for key, val in list(objekt.items()):
            if isdefined(val):
                out[key] = self._clean_container(val, undefinedval)
            elif not skipundefined:
                out[key] = undefinedval
    elif isinstance(objekt, TraitListObject) or isinstance(objekt, list) or isinstance(objekt, tuple):
        out = []
        for val in objekt:
            if isdefined(val):
                out.append(self._clean_container(val, undefinedval))
            elif not skipundefined:
                out.append(undefinedval)
            else:
                out.append(None)
        if isinstance(objekt, tuple):
            out = tuple(out)
    else:
        out = None
        if isdefined(objekt):
            out = objekt
        elif not skipundefined:
            out = undefinedval
    return out