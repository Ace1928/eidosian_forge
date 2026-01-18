from param.parameterized import get_occupied_slots
from .util import datetime_types
def upperexclusive_contains(self, x, y):
    """
        Returns true if the given point is contained within the
        bounding box, where the right and upper boundaries
        are exclusive, and the left and lower boundaries are
        inclusive.  Useful for tiling a plane into non-overlapping
        regions.
        """
    left, bottom, right, top = self.aarect().lbrt()
    return left <= x < right and bottom <= y < top