import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.coords import CoordinateSequence

        Test equals predicate functions correctly regardless of the order
        of the inputs. See issue #317.
        