import numpy as np
from scipy.spatial import cKDTree, distance
def _ensure_spacing(coord, spacing, p_norm, max_out):
    """Returns a subset of coord where a minimum spacing is guaranteed.

    Parameters
    ----------
    coord : ndarray
        The coordinates of the considered points.
    spacing : float
        the maximum allowed spacing between the points.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.
    max_out: int
        If not None, at most the first ``max_out`` candidates are
        returned.

    Returns
    -------
    output : ndarray
        A subset of coord where a minimum spacing is guaranteed.

    """
    tree = cKDTree(coord)
    indices = tree.query_ball_point(coord, r=spacing, p=p_norm)
    rejected_peaks_indices = set()
    naccepted = 0
    for idx, candidates in enumerate(indices):
        if idx not in rejected_peaks_indices:
            candidates.remove(idx)
            dist = distance.cdist([coord[idx]], coord[candidates], distance.minkowski, p=p_norm).reshape(-1)
            candidates = [c for c, d in zip(candidates, dist) if d < spacing]
            rejected_peaks_indices.update(candidates)
            naccepted += 1
            if max_out is not None and naccepted >= max_out:
                break
    output = np.delete(coord, tuple(rejected_peaks_indices), axis=0)
    if max_out is not None:
        output = output[:max_out]
    return output