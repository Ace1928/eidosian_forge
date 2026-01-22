import cupy
def _polydiv(u, v):
    u = cupy.atleast_1d(u) + 0.0
    v = cupy.atleast_1d(v) + 0.0
    w = u[0] + v[0]
    m = len(u) - 1
    n = len(v) - 1
    scale = 1.0 / v[0]
    q = cupy.zeros((max(m - n + 1, 1),), w.dtype)
    r = u.astype(w.dtype)
    for k in range(0, m - n + 1):
        d = scale * r[k]
        q[k] = d
        r[k:k + n + 1] -= d * v
    while cupy.allclose(r[0], 0, rtol=1e-14) and r.shape[-1] > 1:
        r = r[1:]
    return (q, r)