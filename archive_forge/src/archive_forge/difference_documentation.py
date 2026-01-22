import numpy as np
from scipy import sparse
from pygsp import utils
Compute the divergence of a graph signal.

        The divergence of a signal :math:`s` is defined as

        .. math:: y = D^T s,

        where :math:`D` is the differential operator :attr:`D`.

        Parameters
        ----------
        s : ndarray
            Signal of length G.Ne/2 living on the edges (non-directed graph).

        Returns
        -------
        s_div : ndarray
            Divergence signal of length G.N living on the nodes.

        See also
        --------
        compute_differential_operator
        grad : compute the gradient

        Examples
        --------
        >>> G = graphs.Logo()
        >>> G.N, G.Ne
        (1130, 3131)
        >>> s = np.random.normal(size=G.Ne)
        >>> s_div = G.div(s)
        >>> s_grad = G.grad(s_div)

        