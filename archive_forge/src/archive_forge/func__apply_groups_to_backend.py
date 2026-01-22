import inspect
import os
import shutil
import sys
from collections import defaultdict
from inspect import Parameter, Signature
from pathlib import Path
from types import FunctionType
import param
from pyviz_comms import extension as _pyviz_extension
from ..core import (
from ..core.operation import Operation, OperationCallable
from ..core.options import Keywords, Options, options_policy
from ..core.overlay import Overlay
from ..core.util import merge_options_to_dict
from ..operation.element import function
from ..streams import Params, Stream, streams_list_from_dict
from .settings import OutputSettings, list_backends, list_formats
@classmethod
def _apply_groups_to_backend(cls, obj, options, backend, clone):
    """Apply the groups to a single specified backend"""
    obj_handle = obj
    if options is None:
        if clone:
            obj_handle = obj.map(lambda x: x.clone(id=None))
        else:
            obj.map(lambda x: setattr(x, 'id', None))
    elif clone:
        obj_handle = obj.map(lambda x: x.clone(id=x.id))
    return StoreOptions.set_options(obj_handle, options, backend=backend)