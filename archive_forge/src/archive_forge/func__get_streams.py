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
def _get_streams(self, map_obj, watch=True):
    """
        Generates a list of streams to attach to the returned DynamicMap.
        If the input is a DynamicMap any streams that are supplying values
        for the key dimension of the input are inherited. And the list
        of supplied stream classes and instances are processed and
        added to the list.
        """
    if isinstance(self.p.streams, dict):
        streams = defaultdict(dict)
        stream_specs, params = ([], {})
        for name, p in self.p.streams.items():
            if not isinstance(p, param.Parameter):
                raise ValueError('Stream dictionary must map operation keywords to parameter names. Cannot handle %r type.' % type(p))
            if inspect.isclass(p.owner) and issubclass(p.owner, Stream):
                if p.name != name:
                    streams[p.owner][p.name] = name
                else:
                    streams[p.owner] = {}
            else:
                params[name] = p
        stream_specs = streams_list_from_dict(params)
        stream_specs += [stream(rename=rename) for stream, rename in streams.items()]
    else:
        stream_specs = self.p.streams
    streams = []
    op = self.p.operation
    for stream in stream_specs:
        if inspect.isclass(stream) and issubclass(stream, Stream):
            stream = stream()
        elif not (isinstance(stream, Stream) or util.is_param_method(stream)):
            raise ValueError('Streams must be Stream classes or instances, found %s type' % type(stream).__name__)
        if isinstance(op, Operation):
            updates = {k: op.p.get(k) for k, v in stream.contents.items() if v is None and k in op.p}
            if not isinstance(stream, Params):
                reverse = {v: k for k, v in stream._rename.items()}
                updates = {reverse.get(k, k): v for k, v in updates.items()}
            stream.update(**updates)
        streams.append(stream)
    params = {}
    for k, v in self.p.kwargs.items():
        if 'panel' in sys.modules:
            from panel.widgets.base import Widget
            if isinstance(v, Widget):
                v = v.param.value
        if isinstance(v, param.Parameter) and isinstance(v.owner, param.Parameterized):
            params[k] = v
    streams += Params.from_params(params)
    if isinstance(map_obj, DynamicMap):
        dim_streams = util.dimensioned_streams(map_obj)
        streams = list(util.unique_iterator(streams + dim_streams))
    has_dependencies = util.is_param_method(op, has_deps=True) or (isinstance(op, FunctionType) and hasattr(op, '_dinfo'))
    if has_dependencies and watch:
        streams.append(op)
    for value in self.p.kwargs.values():
        if util.is_param_method(value, has_deps=True):
            streams.append(value)
        elif isinstance(value, FunctionType) and hasattr(value, '_dinfo'):
            dependencies = list(value._dinfo.get('dependencies', []))
            dependencies += list(value._dinfo.get('kw', {}).values())
            params = [d for d in dependencies if isinstance(d, param.Parameter) and isinstance(d.owner, param.Parameterized)]
            streams.append(Params(parameters=params, watch_only=True))
    valid, invalid = Stream._process_streams(streams)
    if invalid:
        msg = 'The supplied streams list contains objects that are not Stream instances: {objs}'
        raise TypeError(msg.format(objs=', '.join((f'{el!r}' for el in invalid))))
    return valid