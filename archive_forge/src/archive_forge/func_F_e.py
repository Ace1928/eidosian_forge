import sys
def F_e(x=None, z=None):
    if x is None:
        return (mnl + 1, [x0, 0.0])
    else:
        if z is None:
            v = F(x[0])
            if v is None or v[0] is None:
                return (None, None)
            val = matrix(v[0], tc='d')
            val[0] -= x[1]
            Df = v[1]
        else:
            val, Df, H = F(x[0], z)
            val = matrix(val, tc='d')
            val[0] -= x[1]
        if type(Df) in (matrix, spmatrix):

            def Df_e(u, v, alpha=1.0, beta=0.0, trans='N'):
                if trans == 'N':
                    base.gemv(Df, u[0], v, alpha=alpha, beta=beta, trans='N')
                    v[0] -= alpha * u[1]
                else:
                    base.gemv(Df, u, v[0], alpha=alpha, beta=beta, trans='T')
                    v[1] = -alpha * u[0] + beta * v[1]
        else:

            def Df_e(u, v, alpha=1.0, beta=0.0, trans='N'):
                if trans == 'N':
                    Df(u[0], v, alpha=alpha, beta=beta, trans='N')
                    v[0] -= alpha * u[1]
                else:
                    Df(u, v[0], alpha=alpha, beta=beta, trans='T')
                    v[1] = -alpha * u[0] + beta * v[1]
        if z is None:
            return (val, Df_e)
        else:
            if type(H) in (matrix, spmatrix):

                def H_e(u, v, alpha=1.0, beta=1.0):
                    base.symv(H, u[0], v[0], alpha=alpha, beta=beta)
                    v[1] += beta * v[1]
            else:

                def H_e(u, v, alpha=1.0, beta=1.0):
                    H(u[0], v[0], alpha=alpha, beta=beta)
                    v[1] += beta * v[1]
            return (val, Df_e, H_e)