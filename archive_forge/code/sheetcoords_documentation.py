import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types

        Convert an iterable slicespec (supplying r1,r2,c1,c2 of a
        Slice) into a BoundingRegion specification.

        Exact inverse of _boundsspec2slicespec().
        