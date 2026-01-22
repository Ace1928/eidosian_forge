import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
class BaseDataValidator(BaseValidator):

    def __init__(self, class_strs_map, plotly_name, parent_name, set_uid=False, **kwargs):
        super(BaseDataValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)
        self.class_strs_map = class_strs_map
        self._class_map = {}
        self.set_uid = set_uid

    def description(self):
        trace_types = str(list(self.class_strs_map.keys()))
        trace_types_wrapped = '\n'.join(textwrap.wrap(trace_types, initial_indent='            One of: ', subsequent_indent=' ' * 21, width=79 - 12))
        desc = "    The '{plotly_name}' property is a tuple of trace instances\n    that may be specified as:\n      - A list or tuple of trace instances\n        (e.g. [Scatter(...), Bar(...)])\n      - A single trace instance\n        (e.g. Scatter(...), Bar(...), etc.)\n      - A list or tuple of dicts of string/value properties where:\n        - The 'type' property specifies the trace type\n{trace_types}\n\n        - All remaining properties are passed to the constructor of\n          the specified trace type\n\n        (e.g. [{{'type': 'scatter', ...}}, {{'type': 'bar, ...}}])".format(plotly_name=self.plotly_name, trace_types=trace_types_wrapped)
        return desc

    def get_trace_class(self, trace_name):
        if trace_name not in self._class_map:
            trace_module = import_module('plotly.graph_objs')
            trace_class_name = self.class_strs_map[trace_name]
            self._class_map[trace_name] = getattr(trace_module, trace_class_name)
        return self._class_map[trace_name]

    def validate_coerce(self, v, skip_invalid=False, _validate=True):
        from plotly.basedatatypes import BaseTraceType
        from plotly.graph_objs import Histogram2dcontour
        if v is None:
            v = []
        else:
            if not isinstance(v, (list, tuple)):
                v = [v]
            res = []
            invalid_els = []
            for v_el in v:
                if isinstance(v_el, BaseTraceType):
                    if isinstance(v_el, Histogram2dcontour):
                        v_el = dict(type='histogram2dcontour', **v_el._props)
                    else:
                        v_el = v_el._props
                if isinstance(v_el, dict):
                    type_in_v_el = 'type' in v_el
                    trace_type = v_el.pop('type', 'scatter')
                    if trace_type not in self.class_strs_map:
                        if skip_invalid:
                            trace = self.get_trace_class('scatter')(skip_invalid=skip_invalid, _validate=_validate, **v_el)
                            res.append(trace)
                        else:
                            res.append(None)
                            invalid_els.append(v_el)
                    else:
                        trace = self.get_trace_class(trace_type)(skip_invalid=skip_invalid, _validate=_validate, **v_el)
                        res.append(trace)
                    if type_in_v_el:
                        v_el['type'] = trace_type
                elif skip_invalid:
                    trace = self.get_trace_class('scatter')()
                    res.append(trace)
                else:
                    res.append(None)
                    invalid_els.append(v_el)
            if invalid_els:
                self.raise_invalid_elements(invalid_els)
            v = to_scalar_or_list(res)
            if self.set_uid:
                for trace in v:
                    trace.uid = str(uuid.uuid4())
        return v