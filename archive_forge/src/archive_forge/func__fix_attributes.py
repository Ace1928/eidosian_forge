from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray.backends.common import (
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import (
from xarray.core.variable import Variable
from xarray.namedarray.pycompat import integer_types
def _fix_attributes(attributes):
    attributes = dict(attributes)
    for k in list(attributes):
        if k.lower() == 'global' or k.lower().endswith('_global'):
            attributes.update(attributes.pop(k))
        elif is_dict_like(attributes[k]):
            attributes.update({f'{k}.{k_child}': v_child for k_child, v_child in attributes.pop(k).items()})
    return attributes