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
def _grouped_backends(cls, options, backend):
    """Group options by backend and filter out output group appropriately"""
    if options is None:
        return [(backend or Store.current_backend, options)]
    dfltdict = defaultdict(dict)
    for spec, groups in options.items():
        if 'output' not in groups.keys() or len(groups['output']) == 0:
            dfltdict[backend or Store.current_backend][spec.strip()] = groups
        elif set(groups['output'].keys()) - {'backend'}:
            dfltdict[groups['output']['backend']][spec.strip()] = groups
        elif ['backend'] == list(groups['output'].keys()):
            filtered = {k: v for k, v in groups.items() if k != 'output'}
            dfltdict[groups['output']['backend']][spec.strip()] = filtered
        else:
            raise Exception('The output options group must have the backend keyword')
    return [(bk, bk_opts) for bk, bk_opts in dfltdict.items()]