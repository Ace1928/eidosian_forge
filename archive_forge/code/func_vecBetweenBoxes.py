import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
def vecBetweenBoxes(obj1: 'LTComponent', obj2: 'LTComponent') -> Point:
    """A distance function between two TextBoxes.

    Consider the bounding rectangle for obj1 and obj2.
    Return vector between 2 boxes boundaries if they don't overlap, otherwise
    returns vector betweeen boxes centers

             +------+..........+ (x1, y1)
             | obj1 |          :
             +------+www+------+
             :          | obj2 |
    (x0, y0) +..........+------+
    """
    x0, y0 = (min(obj1.x0, obj2.x0), min(obj1.y0, obj2.y0))
    x1, y1 = (max(obj1.x1, obj2.x1), max(obj1.y1, obj2.y1))
    ow, oh = (x1 - x0, y1 - y0)
    iw, ih = (ow - obj1.width - obj2.width, oh - obj1.height - obj2.height)
    if iw < 0 and ih < 0:
        xc1, yc1 = ((obj1.x0 + obj1.x1) / 2, (obj1.y0 + obj1.y1) / 2)
        xc2, yc2 = ((obj2.x0 + obj2.x1) / 2, (obj2.y0 + obj2.y1) / 2)
        return (xc1 - xc2, yc1 - yc2)
    else:
        return (max(0, iw), max(0, ih))