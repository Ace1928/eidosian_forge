import sys
def A_e(u, v, alpha=1.0, beta=0.0, trans='N'):
    if trans == 'N':
        A(u[0], v, alpha=alpha, beta=beta)
    else:
        A(u, v[0], alpha=alpha, beta=beta, trans='T')
        v[1] *= beta