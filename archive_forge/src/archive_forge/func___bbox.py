from struct import pack, unpack, calcsize, error, Struct
import os
import sys
import time
import array
import tempfile
import logging
import io
from datetime import date
import zipfile
def __bbox(self, s):
    x = []
    y = []
    if len(s.points) > 0:
        px, py = list(zip(*s.points))[:2]
        x.extend(px)
        y.extend(py)
    else:
        raise Exception("Cannot create bbox. Expected a valid shape with at least one point. Got a shape of type '%s' and 0 points." % s.shapeType)
    bbox = [min(x), min(y), max(x), max(y)]
    if self._bbox:
        self._bbox = [min(bbox[0], self._bbox[0]), min(bbox[1], self._bbox[1]), max(bbox[2], self._bbox[2]), max(bbox[3], self._bbox[3])]
    else:
        self._bbox = bbox
    return bbox