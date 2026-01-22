import numpy as np
def _continuous_to_discrete_coords(bounds, level, total_bounds):
    """
    Calculates mid points & ranges of geoms and returns
    as discrete coords

    Parameters
    ----------

    bounds : Bounds of each geometry - array

    p : The number of iterations used in constructing the Hilbert curve

    total_bounds : Total bounds of geometries - array

    Returns
    -------
    Discrete two-dimensional numpy array
    Two-dimensional array Array of hilbert distances for each geom

    """
    side_length = 2 ** level - 1
    x_mids = (bounds[:, 0] + bounds[:, 2]) / 2.0
    y_mids = (bounds[:, 1] + bounds[:, 3]) / 2.0
    if total_bounds is None:
        total_bounds = (np.nanmin(x_mids), np.nanmin(y_mids), np.nanmax(x_mids), np.nanmax(y_mids))
    xmin, ymin, xmax, ymax = total_bounds
    x_int = _continuous_to_discrete(x_mids, (xmin, xmax), side_length)
    y_int = _continuous_to_discrete(y_mids, (ymin, ymax), side_length)
    return (x_int, y_int)