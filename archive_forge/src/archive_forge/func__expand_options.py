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
def _expand_options(cls, options, backend=None):
    """
        Validates and expands a dictionaries of options indexed by
        type[.group][.label] keys into separate style, plot, norm and
        output options.

            opts._expand_options({'Image': dict(cmap='viridis', show_title=False)})

        returns

            {'Image': {'plot': dict(show_title=False), 'style': dict(cmap='viridis')}}
        """
    current_backend = Store.current_backend
    if not Store.renderers:
        raise ValueError('No plotting extension is currently loaded. Ensure you load an plotting extension with hv.extension or import it explicitly from holoviews.plotting before applying any options.')
    elif current_backend not in Store.renderers:
        raise ValueError('Currently selected plotting extension {ext} has not been loaded, ensure you load it with hv.extension({ext}) before setting options'.format(ext=repr(current_backend)))
    try:
        backend_options = Store.options(backend=backend or current_backend)
    except KeyError as e:
        raise Exception(f'The {e} backend is not loaded. Please load the backend using hv.extension.') from None
    expanded = {}
    if isinstance(options, list):
        options = merge_options_to_dict(options)
    for objspec, option_values in options.items():
        objtype = objspec.split('.')[0]
        if objtype not in backend_options:
            raise ValueError(f'{objtype} type not found, could not apply options.')
        obj_options = backend_options[objtype]
        expanded[objspec] = {g: {} for g in obj_options.groups}
        for opt, value in option_values.items():
            for g, group_opts in sorted(obj_options.groups.items()):
                if opt in group_opts.allowed_keywords:
                    expanded[objspec][g][opt] = value
                    break
            else:
                valid_options = sorted({keyword for group_opts in obj_options.groups.values() for keyword in group_opts.allowed_keywords})
                cls._options_error(opt, objtype, backend, valid_options)
    return expanded