from . import matrix
def _evaluate_path(coordinate_object, path):
    """
    Given PtolemyCoordinates or CrossRatios (or more generally, any object that
    supports _get_identity_matrix, short_edge, middle_edge, and long_edge) and
    a path, return the product of the matrices returned by the respective
    calls to short_edge, middle_edge, and long_edge.
    """
    m = coordinate_object._get_identity_matrix()
    for edge in path:
        if isinstance(edge, ShortEdge):
            matrix_method = coordinate_object.short_edge
        elif isinstance(edge, MiddleEdge):
            matrix_method = coordinate_object.middle_edge
        elif isinstance(edge, LongEdge):
            matrix_method = coordinate_object.long_edge
        else:
            raise Exception('Edge of unknown type in path')
        m = matrix.matrix_mult(m, matrix_method(*edge.start_point()))
    return m