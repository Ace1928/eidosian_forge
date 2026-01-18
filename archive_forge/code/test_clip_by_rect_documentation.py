import pytest
from shapely.ops import clip_by_rect
from shapely.wkt import dumps as dump_wkt
from shapely.wkt import loads as load_wkt
Polygon overlapping rectangle