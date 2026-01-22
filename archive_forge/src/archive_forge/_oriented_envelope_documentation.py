import math
from itertools import islice
import numpy as np
import shapely
from shapely.affinity import affine_transform

    Computes the oriented envelope (minimum rotated rectangle) that encloses
    an input geometry.

    This is a fallback implementation for GEOS < 3.12 to have the correct
    minimum area behaviour.
    