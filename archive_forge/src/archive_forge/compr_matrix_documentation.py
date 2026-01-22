import numpy as np
import scipy.sparse as sp
Compresses A and b by eliminating redundant rows.

    Identifies rows that are multiples of another row.
    Reduces A and b to C = PA, d = Pb, where P has one
    nonzero per row.

    Parameters
    ----------
    A : SciPy CSR matrix
        The constraints matrix to compress.
    b : NumPy 1D array
        The vector associated with the constraints matrix.
    equil_eps : float, optional
        Standard for considering two numbers equivalent.

    Returns
    -------
    tuple
        The tuple (A, b, P) where A and b are compressed according to P.
    