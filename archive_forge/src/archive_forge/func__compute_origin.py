from . import matrix
def _compute_origin(choose_generators_info):
    """
    Using the info from SnapPy's choose_generators_info, return the vertex
    (0, 1, 2) of the simplex that SnapPy used to compute a spanning tree of
    the dual 1-skeleton.
    """
    tet = [info['index'] for info in choose_generators_info if info.get('generator_path', -1) == -1][0]
    return Vertex(tet, 0, 1, 2)