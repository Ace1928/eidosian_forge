from .geodesic_info import GeodesicInfo
from .geometric_structure import Filling, FillingMatrix
from ..snap.t3mlite import Mcomplex, simplex
from typing import Tuple, Optional, Sequence
class CuspPostDrillInfo:
    """
    Information carried around to be applied after drilling
    the manifold when re-indexing the cusps, re-applying the
    Dehn-fillings or changing the peripheral curve when drilling
    a core curve.

    Note that we store this information sometime on a
    snappy.snap.t3mlite.Vertex as post_drill_info and sometimes
    as a dictionary tet.post_drill_infos on each tetrahedron assigning
    a CuspPostDrillInfo to each vertex of the tetrahedron. The latter
    representation is more redundant as we need to store the same
    CuspPostDrillInfo to each vertex of each tetrahedron belonging to
    the same vertex class - but is also more convenient in certain
    circumstances.
    """

    def __init__(self, index: Optional[int]=None, filling: Filling=(0, 0), peripheral_matrix: Optional[FillingMatrix]=None):
        self.index = index
        self.filling = filling
        self.peripheral_matrix = peripheral_matrix

    def __eq__(self, other):
        """
        Used for debugging.
        """
        return self.index == self.index and self.filling == self.filling and (self.peripheral_matrix == self.peripheral_matrix)