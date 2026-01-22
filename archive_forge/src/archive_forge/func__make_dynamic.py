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
def _make_dynamic(self, hmap, dynamic_fn, streams):
    """
        Accepts a HoloMap and a dynamic callback function creating
        an equivalent DynamicMap from the HoloMap.
        """
    if isinstance(hmap, ViewableElement):
        dmap = DynamicMap(dynamic_fn, streams=streams)
        if isinstance(hmap, Overlay):
            dmap.callback.inputs[:] = list(hmap)
        return dmap
    dim_values = zip(*hmap.data.keys())
    params = util.get_param_values(hmap)
    kdims = [d.clone(values=list(util.unique_iterator(values))) for d, values in zip(hmap.kdims, dim_values)]
    return DynamicMap(dynamic_fn, streams=streams, **dict(params, kdims=kdims))