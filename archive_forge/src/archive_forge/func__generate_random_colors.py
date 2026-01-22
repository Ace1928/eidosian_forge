import math
import numpy as np
from .draw import polygon as draw_polygon, disk as draw_disk, ellipse as draw_ellipse
from .._shared.utils import warn
def _generate_random_colors(num_colors, num_channels, intensity_range, random):
    """Generate an array of random colors.

    Parameters
    ----------
    num_colors : int
        Number of colors to generate.
    num_channels : int
        Number of elements representing color.
    intensity_range : {tuple of tuples of ints, tuple of ints}, optional
        The range of values to sample pixel values from. For grayscale images
        the format is (min, max). For multichannel - ((min, max),) if the
        ranges are equal across the channels, and
        ((min_0, max_0), ... (min_N, max_N)) if they differ.
    random : `numpy.random.Generator`
        The random state to use for random sampling.

    Raises
    ------
    ValueError
        When the `intensity_range` is not in the interval (0, 255).

    Returns
    -------
    colors : array
        An array of shape (num_colors, num_channels), where the values for
        each channel are drawn from the corresponding `intensity_range`.

    """
    if num_channels == 1:
        intensity_range = (intensity_range,)
    elif len(intensity_range) == 1:
        intensity_range = intensity_range * num_channels
    colors = [random.integers(r[0], r[1] + 1, size=num_colors) for r in intensity_range]
    return np.transpose(colors)