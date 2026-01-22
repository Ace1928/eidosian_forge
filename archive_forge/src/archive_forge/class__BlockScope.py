import threading
import copy
import warnings
import re
import json
from collections import OrderedDict, defaultdict
import numpy as np
from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, np_symbol
from ..symbol import Symbol, load_json
from ..ndarray import NDArray
from .. import name as _name
from .parameter import Parameter, ParameterDict, DeferredInitializationError
from .utils import _indent, _brief_print_list, HookHandle
from .utils import _check_same_symbol_type, _check_all_np_ndarrays
from .. import numpy_extension as _mx_npx
from .. import numpy as _mx_np
from .. util import is_np_array, np_shape, np_array
class _BlockScope(object):
    """Scope for collecting child `Block` s."""
    _current = threading.local()

    def __init__(self, block):
        self._block = block
        self._counter = {}
        self._old_scope = None
        self._name_scope = None

    @staticmethod
    def create(prefix, params, hint):
        """Creates prefix and params for new `Block`."""
        current = getattr(_BlockScope._current, 'value', None)
        if current is None:
            if prefix is None:
                if not hasattr(_name.NameManager._current, 'value'):
                    _name.NameManager._current.value = _name.NameManager()
                prefix = _name.NameManager._current.value.get(None, hint) + '_'
            if params is None:
                params = ParameterDict(prefix)
            else:
                params = ParameterDict(params.prefix, params)
            return (prefix, params)
        if prefix is None:
            count = current._counter.get(hint, 0)
            prefix = '%s%d_' % (hint, count)
            current._counter[hint] = count + 1
        if params is None:
            parent = current._block.params
            params = ParameterDict(parent.prefix + prefix, parent._shared)
        else:
            params = ParameterDict(params.prefix, params)
        return (current._block.prefix + prefix, params)

    def __enter__(self):
        if self._block._empty_prefix:
            return self
        self._old_scope = getattr(_BlockScope._current, 'value', None)
        _BlockScope._current.value = self
        self._name_scope = _name.Prefix(self._block.prefix)
        self._name_scope.__enter__()
        return self

    def __exit__(self, ptype, value, trace):
        if self._block._empty_prefix:
            return
        self._name_scope.__exit__(ptype, value, trace)
        self._name_scope = None
        _BlockScope._current.value = self._old_scope