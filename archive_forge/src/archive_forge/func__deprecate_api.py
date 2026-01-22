import os
import contextlib
import functools
import gc
import socket
import sys
import textwrap
import types
import warnings
def _deprecate_api(old_name, new_name, api, next_version, type=FutureWarning):
    msg = _DEPR_MSG.format(old_name, next_version, new_name)

    def wrapper(*args, **kwargs):
        warnings.warn(msg, type)
        return api(*args, **kwargs)
    return wrapper