import cupy
from cupyx.scipy.interpolate._interpolate import PPoly
@staticmethod
def _find_derivatives(x, y):
    y_shape = y.shape
    if y.ndim == 1:
        x = x[:, None]
        y = y[:, None]
    hk = x[1:] - x[:-1]
    mk = (y[1:] - y[:-1]) / hk
    if y.shape[0] == 2:
        dk = cupy.zeros_like(y)
        dk[0] = mk
        dk[1] = mk
        return dk.reshape(y_shape)
    smk = cupy.sign(mk)
    condition = (smk[1:] != smk[:-1]) | (mk[1:] == 0) | (mk[:-1] == 0)
    w1 = 2 * hk[1:] + hk[:-1]
    w2 = hk[1:] + 2 * hk[:-1]
    whmean = (w1 / mk[:-1] + w2 / mk[1:]) / (w1 + w2)
    dk = cupy.zeros_like(y)
    dk[1:-1] = cupy.where(condition, 0.0, 1.0 / whmean)
    dk[0] = PchipInterpolator._edge_case(hk[0], hk[1], mk[0], mk[1])
    dk[-1] = PchipInterpolator._edge_case(hk[-1], hk[-2], mk[-1], mk[-2])
    return dk.reshape(y_shape)