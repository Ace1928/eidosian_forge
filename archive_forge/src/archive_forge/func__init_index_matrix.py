import numpy as np
from .layer import Layer1Q, Layer2Q
@staticmethod
def _init_index_matrix(dim: int) -> np.ndarray:
    """
        Fast multiplication can be implemented by picking up a subset of
        entries in a sparse matrix.

        Args:
            dim: problem dimensionality.

        Returns:
            2d-array of indices for the fast multiplication.
        """
    all_idx = np.arange(dim * dim, dtype=np.int64).reshape(dim, dim)
    idx = np.full((dim // 4, 4 * 4), fill_value=0, dtype=np.int64)
    b = np.full((4, 4), fill_value=0, dtype=np.int64)
    for i in range(0, dim, 4):
        b[:, :] = all_idx[i:i + 4, i:i + 4]
        idx[i // 4, :] = b.T.ravel()
    return idx