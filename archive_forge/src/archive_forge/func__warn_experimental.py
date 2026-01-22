import copy
import functools
import sys
import warnings
import grpc
from grpc._cython import cygrpc as _cygrpc
def _warn_experimental(api_name, stack_offset):
    if api_name not in _EXPERIMENTAL_APIS_USED:
        _EXPERIMENTAL_APIS_USED.add(api_name)
        msg = "'{}' is an experimental API. It is subject to change or ".format(api_name) + 'removal between minor releases. Proceed with caution.'
        warnings.warn(msg, ExperimentalApiWarning, stacklevel=2 + stack_offset)