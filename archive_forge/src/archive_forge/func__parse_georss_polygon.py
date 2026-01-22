from __future__ import generator_stop
from ..util import FeedParserDict
def _parse_georss_polygon(value, swap=True, dims=2):
    try:
        ring = list(_gen_georss_coords(value, swap, dims))
    except (IndexError, ValueError):
        return None
    if len(ring) < 4:
        return None
    return {'type': 'Polygon', 'coordinates': (ring,)}