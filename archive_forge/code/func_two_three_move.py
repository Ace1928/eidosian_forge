from .cusps import CuspPostDrillInfo
from .geometric_structure import compute_r13_planes_for_tet
from .tracing import compute_plane_intersection_param, Endpoint, GeodesicPiece
from .epsilons import compute_epsilon
from . import constants
from . import exceptions
from ..snap.t3mlite import simplex, Perm4, Tetrahedron # type: ignore
from ..matrix import matrix # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from typing import Sequence, Optional, Union, Tuple, List, Dict, Mapping
def two_three_move(given_pieces: Sequence[GeodesicPiece], verified: bool) -> Sequence[GeodesicPiece]:
    """
    piece is assumed to go from a vertex to opposite face.
    The 2-3 move is performed on the two tetrahedra adjacent to that
    face.
    """
    old_tets = [old_piece.tet for old_piece in given_pieces]
    old_tips = [given_pieces[i].endpoints[i].subsimplex for i in range(2)]
    old_shared_faces = [simplex.comp(old_tip) for old_tip in old_tips]
    RF = old_tets[0].O13_matrices[simplex.F0].base_ring()
    id_matrix = matrix.identity(ring=RF, n=4)
    O13_embeddings = [old_tets[0].O13_matrices[old_shared_faces[0]], id_matrix]
    O13_inverse_embeddings = [old_tets[1].O13_matrices[old_shared_faces[1]], id_matrix]
    tip_points = [O13_embeddings[i] * old_tets[i].R13_vertices[old_tips[i]] for i in range(2)]
    for perm in Perm4.A4():
        if perm.image(simplex.V0) == old_tips[0]:
            break
    gluing = old_tets[0].Gluing[old_shared_faces[0]]
    new_to_old_tets = [[p, gluing * p * Perm4((1, 0, 2, 3))] for p in [perm, perm * Perm4((0, 2, 3, 1)), perm * Perm4((0, 3, 1, 2))]]
    new_tets = [Tetrahedron() for i in range(3)]
    for i, new_tet in enumerate(new_tets):
        new_tet.geodesic_pieces = []
        new_tet.O13_matrices = {simplex.F2: id_matrix, simplex.F3: id_matrix}
        new_tet.PeripheralCurves = [[{v: {face: 0 for face in simplex.TwoSubsimplices} for v in simplex.ZeroSubsimplices} for sheet in range(2)] for ml in range(2)]
        for j, old_tet in enumerate(old_tets):
            new_face = simplex.TwoSubsimplices[1 - j]
            old_face = new_to_old_tets[i][j].image(new_face)
            neighbor = old_tet.Neighbor[old_face]
            new_tet.attach(new_face, neighbor, old_tet.Gluing[old_face] * new_to_old_tets[i][j])
            new_tet.O13_matrices[new_face] = old_tet.O13_matrices[old_face] * O13_inverse_embeddings[j]
            neighbor_face = new_tet.Gluing[new_face].image(new_face)
            neighbor.O13_matrices[neighbor_face] = O13_embeddings[j] * neighbor.O13_matrices[neighbor_face]
            for ml in range(2):
                for sheet in range(2):
                    for v in [simplex.V2, simplex.V3]:
                        old_v = new_to_old_tets[i][j].image(v)
                        new_tet.PeripheralCurves[ml][sheet][v][new_face] = old_tet.PeripheralCurves[ml][sheet][old_v][old_face]
        for ml in range(2):
            for sheet in range(2):
                for v, f in [(simplex.V2, simplex.F3), (simplex.V3, simplex.F2)]:
                    p = new_tet.PeripheralCurves[ml][sheet][v]
                    p[f] = -(p[simplex.F0] + p[simplex.F1])
        new_tet.attach(simplex.F2, new_tets[(i + 1) % 3], (0, 1, 3, 2))
        new_tet.R13_vertices = {simplex.V0: tip_points[0], simplex.V1: tip_points[1], simplex.V2: old_tets[1].R13_vertices[new_to_old_tets[i][1].image(simplex.V2)], simplex.V3: old_tets[1].R13_vertices[new_to_old_tets[i][1].image(simplex.V3)]}
        new_tet.post_drill_infos = {simplex.V0: old_tets[0].post_drill_infos[old_tips[0]], simplex.V1: old_tets[1].post_drill_infos[old_tips[1]], simplex.V2: old_tets[1].post_drill_infos[new_to_old_tets[i][1].image(simplex.V2)], simplex.V3: old_tets[1].post_drill_infos[new_to_old_tets[i][1].image(simplex.V3)]}
        compute_r13_planes_for_tet(new_tet)
    old_to_new_tets = [[~new_to_old_tets[j][i] for j in range(3)] for i in range(2)]
    for j, old_tet in enumerate(old_tets):
        for old_piece in old_tet.geodesic_pieces:
            if old_piece in given_pieces:
                continue
            start_subsimplex = old_piece.endpoints[0].subsimplex
            end_subsimplex = old_piece.endpoints[1].subsimplex
            if start_subsimplex | end_subsimplex in simplex.OneSubsimplices:
                for i, new_tet in enumerate(new_tets):
                    new_start_subsimplex = old_to_new_tets[j][i].image(start_subsimplex)
                    new_end_subsimplex = old_to_new_tets[j][i].image(end_subsimplex)
                    if new_start_subsimplex | new_end_subsimplex == simplex.E23:
                        GeodesicPiece.replace_by(old_piece, old_piece, [GeodesicPiece.create_and_attach(old_piece.index, new_tet, [Endpoint(new_tet.R13_vertices[v], v) for v in [new_start_subsimplex, new_end_subsimplex]])])
                        break
                else:
                    raise Exception('Unhandled edge case.')
                continue
            if start_subsimplex == old_shared_faces[j]:
                continue
            old_pieces = [old_piece]
            if end_subsimplex == old_shared_faces[j]:
                old_pieces.append(old_piece.next_)
                end_j = 1 - j
                end_subsimplex = old_pieces[-1].endpoints[1].subsimplex
            else:
                end_j = j
            r13_endpoints = [O13_embeddings[j] * old_pieces[0].endpoints[0].r13_point, O13_embeddings[end_j] * old_pieces[-1].endpoints[1].r13_point]
            end_cell_dimension = 2
            retrace_direction = +1
            start_j = j
            if end_subsimplex in simplex.ZeroSubsimplices:
                end_cell_dimension = 0
            elif end_subsimplex == simplex.T:
                end_cell_dimension = 3
            elif start_subsimplex == simplex.T:
                end_cell_dimension = 3
                retrace_direction = -1
                start_j, end_j = (end_j, start_j)
                start_subsimplex, end_subsimplex = (end_subsimplex, start_subsimplex)
                r13_endpoints = r13_endpoints[::-1]
            elif not (start_subsimplex in simplex.TwoSubsimplices and end_subsimplex in simplex.TwoSubsimplices):
                raise Exception('Unhandled case')
            for i, new_tet in enumerate(new_tets):
                new_face = simplex.TwoSubsimplices[1 - start_j]
                new_start_subsimplex = old_to_new_tets[start_j][i].image(start_subsimplex)
                if new_start_subsimplex == new_face:
                    GeodesicPiece.replace_by(old_pieces[0], old_pieces[-1], _retrace_geodesic_piece(old_piece.index, new_tets, new_tet, new_face, end_cell_dimension, r13_endpoints, retrace_direction, verified, allowed_end_corners=None))
                    break
            else:
                raise Exception('No match')
    new_piece = GeodesicPiece.create_and_attach(given_pieces[0].index, new_tets[0], [Endpoint(tip_points[0], simplex.V0), Endpoint(tip_points[1], simplex.V1)])
    GeodesicPiece.replace_by(given_pieces[0], given_pieces[1], [new_piece])
    return new_piece