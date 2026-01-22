import numpy as np
def GPA(A, ff=None, vgQ=None, T=None, max_tries=501, rotation_method='orthogonal', tol=1e-05):
    """
    The gradient projection algorithm (GPA) minimizes a target function
    :math:`\\phi(L)`, where :math:`L` is a matrix with rotated factors.

    For orthogonal rotation methods :math:`L=AT`, where :math:`T` is an
    orthogonal matrix. For oblique rotation matrices :math:`L=A(T^*)^{-1}`,
    where :math:`T` is a normal matrix, i.e., :math:`TT^*=T^*T`. Oblique
    rotations relax the orthogonality constraint in order to gain simplicity
    in the interpretation.

    Parameters
    ----------
    A : numpy matrix
        non rotated factors
    T : numpy matrix (default identity matrix)
        initial guess of rotation matrix
    ff : function (defualt None)
        criterion :math:`\\phi` to optimize. Should have A, T, L as keyword
        arguments
        and mapping to a float. Only used (and required) if vgQ is not
        provided.
    vgQ : function (defualt None)
        criterion :math:`\\phi` to optimize and its derivative. Should have
         A, T, L as keyword arguments and mapping to a tuple containing a
        float and vector. Can be omitted if ff is provided.
    max_tries : int (default 501)
        maximum number of iterations
    rotation_method : str
        should be one of {orthogonal, oblique}
    tol : float
        stop criterion, algorithm stops if Frobenius norm of gradient is
        smaller then tol
    """
    if rotation_method not in ['orthogonal', 'oblique']:
        raise ValueError('rotation_method should be one of {orthogonal, oblique}')
    if vgQ is None:
        if ff is None:
            raise ValueError('ff should be provided if vgQ is not')
        derivative_free = True
        Gff = lambda x: Gf(x, lambda y: ff(T=y, A=A, L=None))
    else:
        derivative_free = False
    if T is None:
        T = np.eye(A.shape[1])
    al = 1
    table = []
    if derivative_free:
        f = ff(T=T, A=A, L=None)
        G = Gff(T)
    elif rotation_method == 'orthogonal':
        L = A.dot(T)
        f, Gq = vgQ(L=L)
        G = A.T.dot(Gq)
    else:
        Ti = np.linalg.inv(T)
        L = A.dot(Ti.T)
        f, Gq = vgQ(L=L)
        G = -L.T.dot(Gq).dot(Ti).T
    for i_try in range(0, max_tries):
        if rotation_method == 'orthogonal':
            M = T.T.dot(G)
            S = (M + M.T) / 2
            Gp = G - T.dot(S)
        else:
            Gp = G - T.dot(np.diag(np.sum(T * G, axis=0)))
        s = np.linalg.norm(Gp, 'fro')
        table.append([i_try, f, np.log10(s), al])
        if s < tol:
            break
        al = 2 * al
        for i in range(11):
            X = T - al * Gp
            if rotation_method == 'orthogonal':
                U, D, V = np.linalg.svd(X, full_matrices=False)
                Tt = U.dot(V)
            else:
                v = 1 / np.sqrt(np.sum(X ** 2, axis=0))
                Tt = X.dot(np.diag(v))
            if derivative_free:
                ft = ff(T=Tt, A=A, L=None)
            elif rotation_method == 'orthogonal':
                L = A.dot(Tt)
                ft, Gq = vgQ(L=L)
            else:
                Ti = np.linalg.inv(Tt)
                L = A.dot(Ti.T)
                ft, Gq = vgQ(L=L)
            if ft < f - 0.5 * s ** 2 * al:
                break
            al = al / 2
        T = Tt
        f = ft
        if derivative_free:
            G = Gff(T)
        elif rotation_method == 'orthogonal':
            G = A.T.dot(Gq)
        else:
            G = -L.T.dot(Gq).dot(Ti).T
    Th = T
    Lh = rotateA(A, T, rotation_method=rotation_method)
    Phi = T.T.dot(T)
    return (Lh, Phi, Th, table)