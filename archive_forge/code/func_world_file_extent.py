import collections
from pathlib import Path
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
@staticmethod
def world_file_extent(worldfile_handle, im_shape):
    """
        Return the extent ``(x0, x1, y0, y1)`` and pixel size
        ``(x_width, y_width)`` as defined in the given worldfile file handle
        and associated image shape ``(x, y)``.

        """
    lines = worldfile_handle.readlines()
    if len(lines) != 6:
        raise ValueError('Only world files with 6 lines are supported.')
    pix_size = (float(lines[0]), float(lines[3]))
    pix_rotation = (float(lines[1]), float(lines[2]))
    if pix_rotation != (0.0, 0.0):
        raise ValueError('Rotated pixels in world files is not currently supported.')
    ul_corner = (float(lines[4]), float(lines[5]))
    min_x, max_x = (ul_corner[0] - pix_size[0] / 2.0, ul_corner[0] + pix_size[0] * im_shape[0] - pix_size[0] / 2.0)
    min_y, max_y = (ul_corner[1] - pix_size[1] / 2.0, ul_corner[1] + pix_size[1] * im_shape[1] - pix_size[1] / 2.0)
    return ((min_x, max_x, min_y, max_y), pix_size)