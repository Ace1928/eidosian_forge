from weakref import WeakValueDictionary
from ...element import Tiles
from ...streams import (
from .util import _trace_to_subplot
class BoundsCallback(PlotlyCallback):
    callback_properties = ['selected_data']
    boundsx = False
    boundsy = False

    @classmethod
    def get_event_data_from_property_update(cls, property, selected_data, fig_dict):
        traces = fig_dict.get('data', [])
        event_data = {}
        for trace in traces:
            trace_uid = trace.get('uid', None)
            if cls.boundsx and cls.boundsy:
                stream_data = dict(bounds=None)
            elif cls.boundsx:
                stream_data = dict(boundsx=None)
            elif cls.boundsy:
                stream_data = dict(boundsy=None)
            else:
                stream_data = {}
            event_data[trace_uid] = stream_data
        range_data = (selected_data or {}).get('range', {})
        cls.update_event_data_xyaxis(range_data, traces, event_data)
        cls.update_event_data_mapbox(range_data, traces, event_data)
        return event_data

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

    @classmethod
    def update_event_data_mapbox(cls, range_data, traces, event_data):
        for trace in traces:
            trace_type = trace.get('type', 'scatter')
            trace_uid = trace.get('uid', None)
            if _trace_to_subplot.get(trace_type, None) != ['mapbox']:
                continue
            mapbox_ref = trace.get('subplot', 'mapbox')
            if mapbox_ref in range_data:
                lon_bounds = [range_data[mapbox_ref][0][0], range_data[mapbox_ref][1][0]]
                lat_bounds = [range_data[mapbox_ref][0][1], range_data[mapbox_ref][1][1]]
                easting, northing = Tiles.lon_lat_to_easting_northing(lon_bounds, lat_bounds)
                new_bounds = (easting[0], northing[0], easting[1], northing[1])
                if cls.boundsx and cls.boundsy:
                    stream_data = dict(bounds=new_bounds)
                elif cls.boundsx:
                    stream_data = dict(boundsx=(new_bounds[0], new_bounds[2]))
                elif cls.boundsy:
                    stream_data = dict(boundsy=(new_bounds[1], new_bounds[3]))
                else:
                    stream_data = {}
                event_data[trace_uid] = stream_data