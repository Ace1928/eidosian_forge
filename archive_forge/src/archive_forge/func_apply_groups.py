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
def apply_groups(cls, obj, options=None, backend=None, clone=True, **kwargs):
    """Applies nested options definition grouped by type.

        Applies options on an object or nested group of objects,
        returning a new object with the options applied. This method
        accepts the separate option namespaces explicitly (i.e. 'plot',
        'style', and 'norm').

        If the options are to be set directly on the object a
        simple format may be used, e.g.:

            opts.apply_groups(obj, style={'cmap': 'viridis'},
                                         plot={'show_title': False})

        If the object is nested the options must be qualified using
        a type[.group][.label] specification, e.g.:

            opts.apply_groups(obj, {'Image': {'plot':  {'show_title': False},
                                              'style': {'cmap': 'viridis}}})

        If no opts are supplied all options on the object will be reset.

        Args:
            options (dict): Options specification
                Options specification should be indexed by
                type[.group][.label] or option type ('plot', 'style',
                'norm').
            backend (optional): Backend to apply options to
                Defaults to current selected backend
            clone (bool, optional): Whether to clone object
                Options can be applied inplace with clone=False
            **kwargs: Keywords of options by type
                Applies options directly to the object by type
                (e.g. 'plot', 'style', 'norm') specified as
                dictionaries.

        Returns:
            Returns the object or a clone with the options applied
        """
    if isinstance(options, str):
        from ..util.parser import OptsSpec
        try:
            options = OptsSpec.parse(options)
        except SyntaxError:
            options = OptsSpec.parse(f'{obj.__class__.__name__} {options}')
    if kwargs:
        options = cls._group_kwargs_to_options(obj, kwargs)
    for backend_loop, backend_opts in cls._grouped_backends(options, backend):
        obj = cls._apply_groups_to_backend(obj, backend_opts, backend_loop, clone)
    return obj