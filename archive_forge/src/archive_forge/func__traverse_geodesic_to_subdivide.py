from .moves import one_four_move, two_three_move
from .tracing import GeodesicPiece, GeodesicPieceTracker
from .exceptions import GeodesicStartingPiecesCrossSameFaceError
from . import debug
from ..snap.t3mlite import Mcomplex, Tetrahedron
from typing import Sequence, Dict
def _traverse_geodesic_to_subdivide(start_piece: GeodesicPiece, verified: bool) -> GeodesicPiece:
    debug.check_consistency_2(start_piece)
    if start_piece.prev.endpoints[0].subsimplex == start_piece.endpoints[1].subsimplex:
        raise GeodesicStartingPiecesCrossSameFaceError()
    end_piece, piece = one_four_move([start_piece.prev, start_piece], verified)
    debug.check_consistency_2(piece)
    while True:
        piece = piece.next_
        if piece.is_face_to_vertex():
            piece = two_three_move([piece.prev, piece], verified)
            debug.check_consistency_2(piece)
            return piece
        piece, next_piece = one_four_move([piece], verified)
        debug.check_consistency_2(piece)
        piece = two_three_move([piece.prev, piece], verified)
        debug.check_consistency_2(piece)
        piece = piece.next_