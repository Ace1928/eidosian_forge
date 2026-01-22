from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
@staticmethod
def _generic_layout(diagram, merged_morphisms):
    """
        Produces the generic layout for the supplied diagram.
        """
    all_objects = set(diagram.objects)
    if len(all_objects) == 1:
        grid = _GrowableGrid(1, 1)
        grid[0, 0] = tuple(all_objects)[0]
        return grid
    skeleton = DiagramGrid._build_skeleton(merged_morphisms)
    grid = _GrowableGrid(2, 1)
    if len(skeleton) == 1:
        objects = sorted(all_objects, key=default_sort_key)
        grid[0, 0] = objects[0]
        grid[0, 1] = objects[1]
        return grid
    triangles = DiagramGrid._list_triangles(skeleton)
    triangles = DiagramGrid._drop_redundant_triangles(triangles, skeleton)
    triangle_sizes = DiagramGrid._compute_triangle_min_sizes(triangles, skeleton)
    triangles = sorted(triangles, key=lambda tri: DiagramGrid._triangle_key(tri, triangle_sizes))
    root_edge = DiagramGrid._pick_root_edge(triangles[0], skeleton)
    grid[0, 0], grid[0, 1] = root_edge
    fringe = [((0, 0), (0, 1))]
    placed_objects = set(root_edge)
    while placed_objects != all_objects:
        welding = DiagramGrid._find_triangle_to_weld(triangles, fringe, grid)
        if welding:
            triangle, welding_edge = welding
            restart_required = DiagramGrid._weld_triangle(triangle, welding_edge, fringe, grid, skeleton)
            if restart_required:
                continue
            placed_objects.update(DiagramGrid._triangle_objects(triangle))
        else:
            new_obj = DiagramGrid._grow_pseudopod(triangles, fringe, grid, skeleton, placed_objects)
            if not new_obj:
                remaining_objects = all_objects - placed_objects
                remaining_diagram = diagram.subdiagram_from_objects(FiniteSet(*remaining_objects))
                remaining_grid = DiagramGrid(remaining_diagram)
                final_width = grid.width + remaining_grid.width
                final_height = max(grid.height, remaining_grid.height)
                final_grid = _GrowableGrid(final_width, final_height)
                for i in range(grid.width):
                    for j in range(grid.height):
                        final_grid[i, j] = grid[i, j]
                start_j = grid.width
                for i in range(remaining_grid.height):
                    for j in range(remaining_grid.width):
                        final_grid[i, start_j + j] = remaining_grid[i, j]
                return final_grid
            placed_objects.add(new_obj)
        triangles = DiagramGrid._drop_irrelevant_triangles(triangles, placed_objects)
    return grid