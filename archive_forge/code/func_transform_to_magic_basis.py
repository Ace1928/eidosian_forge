from __future__ import annotations
import numpy as np
def transform_to_magic_basis(U: np.ndarray, reverse: bool=False) -> np.ndarray:
    """Transform the 4-by-4 matrix ``U`` into the magic basis.

    This method internally uses non-normalized versions of the basis to minimize the floating-point
    errors that arise during the transformation.

    Args:
        U (np.ndarray): 4-by-4 matrix to transform.
        reverse (bool): Whether to do the transformation forwards (``B @ U @ B.conj().T``, the
        default) or backwards (``B.conj().T @ U @ B``).

    Returns:
        np.ndarray: The transformed 4-by-4 matrix.
    """
    if reverse:
        return _B_nonnormalized_dagger @ U @ _B_nonnormalized
    return _B_nonnormalized @ U @ _B_nonnormalized_dagger