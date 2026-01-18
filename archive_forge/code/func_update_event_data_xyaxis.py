from weakref import WeakValueDictionary
from ...element import Tiles
from ...streams import (
from .util import _trace_to_subplot
@classmethod
def update_event_data_xyaxis(cls, range_data, traces, event_data):
    for trace in traces:
        trace_type = trace.get('type', 'scatter')
        trace_uid = trace.get('uid', None)
        if _trace_to_subplot.get(trace_type, None) != ['xaxis', 'yaxis']:
            continue
        xref = trace.get('xaxis', 'x')
        yref = trace.get('yaxis', 'y')
        if xref in range_data and yref in range_data:
            new_bounds = (range_data[xref][0], range_data[yref][0], range_data[xref][1], range_data[yref][1])
            if cls.boundsx and cls.boundsy:
                stream_data = dict(bounds=new_bounds)
            elif cls.boundsx:
                stream_data = dict(boundsx=(new_bounds[0], new_bounds[2]))
            elif cls.boundsy:
                stream_data = dict(boundsy=(new_bounds[1], new_bounds[3]))
            else:
                stream_data = {}
            event_data[trace_uid] = stream_data