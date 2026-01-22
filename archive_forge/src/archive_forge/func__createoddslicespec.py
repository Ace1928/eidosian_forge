import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
@staticmethod
def _createoddslicespec(bounds, scs, min_matrix_radius):
    """
        Create the 'odd' Slice that best approximates the specified
        sheet-coordinate bounds.

        The supplied bounds are translated to have a center at the
        center of one of the sheet's units (we arbitrarily use the
        center unit), and then these bounds are converted to a slice
        in such a way that the slice exactly includes all units whose
        centers are within the bounds (see boundsspec2slicespec()).
        However, to ensure that the bounds are treated symmetrically,
        we take the right and bottom bounds and reflect these about
        the center of the slice (i.e. we take the 'xradius' to be
        right_col-center_col and the 'yradius' to be
        bottom_col-center_row). Hence, if the bounds happen to go
        through units, if the units are included on the right and
        bottom bounds, they will be included on the left and top
        bounds. This ensures that the slice has odd dimensions.
        """
    bounds_xcenter, bounds_ycenter = bounds.centroid()
    sheet_rows, sheet_cols = scs.shape
    center_row, center_col = (sheet_rows / 2, sheet_cols / 2)
    unit_xcenter, unit_ycenter = scs.matrixidx2sheet(center_row, center_col)
    bounds.translate(unit_xcenter - bounds_xcenter, unit_ycenter - bounds_ycenter)
    r1, r2, c1, c2 = Slice._boundsspec2slicespec(bounds.lbrt(), scs)
    xrad = max(c2 - center_col - 1, min_matrix_radius)
    yrad = max(r2 - center_row - 1, min_matrix_radius)
    r2 = center_row + yrad + 1
    c2 = center_col + xrad + 1
    r1 = center_row - yrad
    c1 = center_col - xrad
    return (r1, r2, c1, c2)