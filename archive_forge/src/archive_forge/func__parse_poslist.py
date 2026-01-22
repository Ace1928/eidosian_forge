from __future__ import generator_stop
from ..util import FeedParserDict
def _parse_poslist(value, geom_type, swap=True, dims=2):
    if geom_type == 'linestring':
        return _parse_georss_line(value, swap, dims)
    elif geom_type == 'polygon':
        ring = _parse_georss_line(value, swap, dims)
        return {'type': 'Polygon', 'coordinates': (ring['coordinates'],)}
    else:
        return None