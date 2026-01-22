import os
import weakref
import torch
def _try_get_jit_cached_function(key):
    if getattr(key, '__disable_jit_function_caching__', False) is True:
        return None
    qual_name = _jit_caching_layer.get(key, None)
    if qual_name:
        return _python_cu.find_function(qual_name)
    else:
        return None