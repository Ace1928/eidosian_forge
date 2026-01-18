import numpy as np
def vgQ_target(H, L=None, A=None, T=None, rotation_method='orthogonal'):
    """
    Subroutine for the value of vgQ using orthogonal or oblique rotation
    towards a target matrix, i.e., we minimize:

    .. math::
        \\phi(L) =\\frac{1}{2}\\|L-H\\|^2

    and the gradient is given by

    .. math::
        d\\phi(L)=L-H.

    Either :math:`L` should be provided or :math:`A` and :math:`T` should be
    provided.

    For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix. For oblique rotations :math:`L`
    satisfies

    .. math::
        L =  A(T^*)^{-1},

    where :math:`T` is a normal matrix.

    Parameters
    ----------
    H : numpy matrix
        target matrix
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    rotation_method : str
        should be one of {orthogonal, oblique}
    """
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method=rotation_method)
    q = np.linalg.norm(L - H, 'fro') ** 2
    Gq = 2 * (L - H)
    return (q, Gq)