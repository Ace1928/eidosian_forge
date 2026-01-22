import warnings
from functools import wraps
from typing import Callable
from .. import constants
@wraps(fn)
def _inner_fn(*args, **kwargs):
    if not constants.HF_HUB_DISABLE_EXPERIMENTAL_WARNING:
        warnings.warn(f"'{name}' is experimental and might be subject to breaking changes in the future. You can disable this warning by setting `HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1` as environment variable.", UserWarning)
    return fn(*args, **kwargs)