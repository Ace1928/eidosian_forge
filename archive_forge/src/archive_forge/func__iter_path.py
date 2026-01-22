from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def _iter_path(pointer):
    """Take a cairo_path_t * pointer
    and yield ``(path_operation, coordinates)`` tuples.

    See :meth:`Context.copy_path` for the data structure.

    """
    _check_status(pointer.status)
    data = pointer.data
    num_data = pointer.num_data
    points_per_type = PATH_POINTS_PER_TYPE
    position = 0
    while position < num_data:
        path_data = data[position]
        path_type = path_data.header.type
        points = ()
        for i in range(points_per_type[path_type]):
            point = data[position + i + 1].point
            points += (point.x, point.y)
        yield (path_type, points)
        position += path_data.header.length