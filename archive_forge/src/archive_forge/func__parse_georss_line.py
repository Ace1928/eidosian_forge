from __future__ import generator_stop
from ..util import FeedParserDict
def _parse_georss_line(value, swap=True, dims=2):
    try:
        coords = list(_gen_georss_coords(value, swap, dims))
        return {'type': 'LineString', 'coordinates': coords}
    except (IndexError, ValueError):
        return None