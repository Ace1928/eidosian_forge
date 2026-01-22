from functools import reduce
import numpy as np
from ..draw import polygon
from .._shared.version_requirements import require
def _on_lasso_selection(vertices):
    if len(vertices) < 3:
        return
    list_of_vertex_lists.append(vertices)
    polygon_object = _draw_polygon(ax, vertices, alpha=alpha)
    polygons_drawn.append(polygon_object)
    plt.draw()