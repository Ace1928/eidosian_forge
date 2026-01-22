from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.matrices.dense import eye
from sympy.matrices.expressions.trace import trace
from sympy.tensor.tensor import TensorIndexType, TensorIndex,\
def _trace_single_line1(t):
    t = t.sorted_components()
    components = t.components
    ncomps = len(components)
    g = LorentzIndex.metric
    hit = 0
    for i in range(ncomps):
        if components[i] == GammaMatrix:
            hit = 1
            break
    for j in range(i + hit, ncomps):
        if components[j] != GammaMatrix:
            break
    else:
        j = ncomps
    numG = j - i
    if numG == 0:
        tcoeff = t.coeff
        return t.nocoeff if tcoeff else t
    if numG % 2 == 1:
        return TensMul.from_data(S.Zero, [], [], [])
    elif numG > 4:
        a = t.split()
        ind1 = a[i].get_indices()[0]
        ind2 = a[i + 1].get_indices()[0]
        aa = a[:i] + a[i + 2:]
        t1 = tensor_mul(*aa) * g(ind1, ind2)
        t1 = t1.contract_metric(g)
        args = [t1]
        sign = 1
        for k in range(i + 2, j):
            sign = -sign
            ind2 = a[k].get_indices()[0]
            aa = a[:i] + a[i + 1:k] + a[k + 1:]
            t2 = sign * tensor_mul(*aa) * g(ind1, ind2)
            t2 = t2.contract_metric(g)
            t2 = simplify_gpgp(t2, False)
            args.append(t2)
        t3 = TensAdd(*args)
        t3 = _trace_single_line(t3)
        return t3
    else:
        a = t.split()
        t1 = _gamma_trace1(*a[i:j])
        a2 = a[:i] + a[j:]
        t2 = tensor_mul(*a2)
        t3 = t1 * t2
        if not t3:
            return t3
        t3 = t3.contract_metric(g)
        return t3