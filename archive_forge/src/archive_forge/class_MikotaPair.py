import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import kron, eye, dia_array
class MikotaPair:
    """
    Construct the Mikota pair of matrices in various formats and
    eigenvalues of the generalized eigenproblem with them.

    The Mikota pair of matrices [1, 2]_ models a vibration problem
    of a linear mass-spring system with the ends attached where
    the stiffness of the springs and the masses increase along
    the system length such that vibration frequencies are subsequent
    integers 1, 2, ..., `n` where `n` is the number of the masses. Thus,
    eigenvalues of the generalized eigenvalue problem for
    the matrix pair `K` and `M` where `K` is he system stiffness matrix
    and `M` is the system mass matrix are the squares of the integers,
    i.e., 1, 4, 9, ..., ``n * n``.

    The stiffness matrix `K` is square real tri-diagonal symmetric
    positive definite. The mass matrix `M` is diagonal with diagonal
    entries 1, 1/2, 1/3, ...., ``1/n``. Both matrices get
    ill-conditioned with `n` growing.

    Parameters
    ----------
    n : int
        The size of the matrices of the Mikota pair.
    dtype : dtype
        Numerical type of the array. Default is ``np.float64``.

    Attributes
    ----------
    eigenvalues : 1D ndarray, ``np.uint64``
        All eigenvalues of the Mikota pair ordered ascending.

    Methods
    -------
    MikotaK()
        A `LinearOperator` custom object for the stiffness matrix.
    MikotaM()
        A `LinearOperator` custom object for the mass matrix.
    
    .. versionadded:: 1.12.0

    References
    ----------
    .. [1] J. Mikota, "Frequency tuning of chain structure multibody oscillators
       to place the natural frequencies at omega1 and N-1 integer multiples
       omega2,..., omegaN", Z. Angew. Math. Mech. 81 (2001), S2, S201-S202.
       Appl. Num. Anal. Comp. Math. Vol. 1 No. 2 (2004).
    .. [2] Peter C. Muller and Metin Gurgoze,
       "Natural frequencies of a multi-degree-of-freedom vibration system",
       Proc. Appl. Math. Mech. 6, 319-320 (2006).
       http://dx.doi.org/10.1002/pamm.200610141.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg._special_sparse_arrays import MikotaPair
    >>> n = 6
    >>> mik = MikotaPair(n)
    >>> mik_k = mik.k
    >>> mik_m = mik.m
    >>> mik_k.toarray()
    array([[11., -5.,  0.,  0.,  0.,  0.],
           [-5.,  9., -4.,  0.,  0.,  0.],
           [ 0., -4.,  7., -3.,  0.,  0.],
           [ 0.,  0., -3.,  5., -2.,  0.],
           [ 0.,  0.,  0., -2.,  3., -1.],
           [ 0.,  0.,  0.,  0., -1.,  1.]])
    >>> mik_k.tobanded()
    array([[ 0., -5., -4., -3., -2., -1.],
           [11.,  9.,  7.,  5.,  3.,  1.]])
    >>> mik_m.tobanded()
    array([1.        , 0.5       , 0.33333333, 0.25      , 0.2       ,
        0.16666667])
    >>> mik_k.tosparse()
    <6x6 sparse matrix of type '<class 'numpy.float64'>'
        with 16 stored elements (3 diagonals) in DIAgonal format>
    >>> mik_m.tosparse()
    <6x6 sparse matrix of type '<class 'numpy.float64'>'
        with 6 stored elements (1 diagonals) in DIAgonal format>
    >>> np.array_equal(mik_k(np.eye(n)), mik_k.toarray())
    True
    >>> np.array_equal(mik_m(np.eye(n)), mik_m.toarray())
    True
    >>> mik.eigenvalues()
    array([ 1,  4,  9, 16, 25, 36])  
    >>> mik.eigenvalues(2)
    array([ 1,  4])

    """

    def __init__(self, n, dtype=np.float64):
        self.n = n
        self.dtype = dtype
        self.shape = (n, n)
        self.m = MikotaM(self.shape, self.dtype)
        self.k = MikotaK(self.shape, self.dtype)

    def eigenvalues(self, m=None):
        """Return the requested number of eigenvalues.
        
        Parameters
        ----------
        m : int, optional
            The positive number of smallest eigenvalues to return.
            If not provided, then all eigenvalues will be returned.
            
        Returns
        -------
        eigenvalues : `np.uint64` array
            The requested `m` smallest or all eigenvalues, in ascending order.
        """
        if m is None:
            m = self.n
        arange_plus1 = np.arange(1, m + 1, dtype=np.uint64)
        return arange_plus1 * arange_plus1