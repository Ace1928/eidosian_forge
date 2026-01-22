import spherogram
import random
import itertools
from . import pl_utils
from .rational_linear_algebra import QQ, Matrix, Vector3
from .exceptions import GeneralPositionError
def _setup_crossings(self):
    pts = self.points
    crossings, arcs = ([], [])
    for component in self.components:
        successive_pairs = [(c, component[(i + 1) % len(component)]) for i, c in enumerate(component)]
        arcs += [Arc(pts[i], i, pts[j], j, i) for i, j in successive_pairs]
    for A, B in itertools.combinations(arcs, 2):
        a = (proj(A[0]), proj(A[1]))
        b = (proj(B[0]), proj(B[1]))
        if A.j == B.i:
            continue
        elif B.j == A.i:
            continue
        elif pl_utils.segments_meet_not_at_endpoint(a, b):
            M = Matrix([a[1] - a[0], b[0] - b[1]]).transpose()
            if M.rank() != 2:
                raise GeneralPositionError('Segments overlap on their interiors')
            s, t = M.solve_right(b[0] - a[0])
            e = 1e-12
            if not (e < s < 1 - e and e < t < 1 - e):
                raise GeneralPositionError('Intersection too near the end of one segment')
            x_a = (1 - s) * A[0] + s * A[1]
            x_b = (1 - t) * B[0] + t * B[1]
            assert norm_sq(proj(x_a - x_b)) < 1e-05
            height_a = x_a[2]
            height_b = x_b[2]
            assert abs(height_a - height_b) > 1e-14
            if height_a > height_b:
                crossings.append(Crossing(A, B, s, t, len(crossings)))
            else:
                crossings.append(Crossing(B, A, t, s, len(crossings)))
    self.crossings, self.arcs = (crossings, arcs)