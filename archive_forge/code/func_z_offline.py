import math
from .functions import defun
def z_offline(ctx, w, k=0):
    """
    Computes Z(w) and its derivatives off the line
    """
    s = ctx.mpf('0.5') + ctx.j * w
    s1 = s
    s2 = ctx.conj(1 - s1)
    wpinitial = ctx.prec
    ctx.prec = 35
    if ctx._re(s1) >= 0:
        M1 = 2 * ctx.sqrt(ctx._im(s1) / (2 * ctx.pi))
        X = ctx.sqrt(abs(s1))
    else:
        X = (2 * ctx.pi) ** (ctx._re(s1) - 1) * abs(1 - s1) ** (0.5 - ctx._re(s1))
        M1 = 4 * ctx._im(s1) * X
    if ctx._re(s2) >= 0:
        M2 = 2 * ctx.sqrt(ctx._im(s2) / (2 * ctx.pi))
    else:
        M2 = 4 * ctx._im(s2) * (2 * ctx.pi) ** (ctx._re(s2) - 1) * abs(1 - s2) ** (0.5 - ctx._re(s2))
    T = 2 * abs(ctx.siegeltheta(w))
    aux1 = ctx.sqrt(X)
    aux2 = aux1 * (M1 + M2)
    aux3 = 3 + wpinitial
    wpbasic = max(6, 3 + ctx.mag(T), ctx.mag(aux2 * (26 + 2 * T)) + aux3)
    wptheta = max(4, ctx.mag(2.04 * aux2) + aux3)
    wpR = ctx.mag(4 * aux1) + aux3
    ctx.prec = wptheta
    theta = ctx.siegeltheta(w)
    ctx.prec = wpR
    xrz, yrz = Rzeta_simul(ctx, s, k)
    pta = 0.25 + 0.5j * w
    ptb = 0.25 - 0.5j * w
    if k > 0:
        ps1 = 0.25 * (ctx.psi(0, pta) + ctx.psi(0, ptb)) - ctx.ln(ctx.pi) / 2
    if k > 1:
        ps2 = 1j / 8 * (ctx.psi(1, pta) - ctx.psi(1, ptb))
    if k > 2:
        ps3 = -1.0 / 16 * (ctx.psi(2, pta) + ctx.psi(2, ptb))
    if k > 3:
        ps4 = -1j / 32 * (ctx.psi(3, pta) - ctx.psi(3, ptb))
    ctx.prec = wpbasic
    exptheta = ctx.expj(theta)
    if k == 0:
        zv = exptheta * xrz[0] + yrz[0] / exptheta
    j = ctx.j
    if k == 1:
        zv = j * exptheta * (xrz[1] + xrz[0] * ps1) - j * (yrz[1] + yrz[0] * ps1) / exptheta
    if k == 2:
        zv = exptheta * (-2 * xrz[1] * ps1 - xrz[0] * ps1 ** 2 - xrz[2] + j * xrz[0] * ps2)
        zv = zv + (-2 * yrz[1] * ps1 - yrz[0] * ps1 ** 2 - yrz[2] - j * yrz[0] * ps2) / exptheta
    if k == 3:
        zv1 = -3 * xrz[1] * ps1 ** 2 - xrz[0] * ps1 ** 3 - 3 * xrz[2] * ps1 + j * 3 * xrz[1] * ps2
        zv1 = (zv1 + 3j * xrz[0] * ps1 * ps2 - xrz[3] + xrz[0] * ps3) * j * exptheta
        zv2 = 3 * yrz[1] * ps1 ** 2 + yrz[0] * ps1 ** 3 + 3 * yrz[2] * ps1 + j * 3 * yrz[1] * ps2
        zv2 = j * (zv2 + 3j * yrz[0] * ps1 * ps2 + yrz[3] - yrz[0] * ps3) / exptheta
        zv = zv1 + zv2
    if k == 4:
        zv1 = 4 * xrz[1] * ps1 ** 3 + xrz[0] * ps1 ** 4 + 6 * xrz[2] * ps1 ** 2
        zv1 = zv1 - 12j * xrz[1] * ps1 * ps2 - 6j * xrz[0] * ps1 ** 2 * ps2 - 6j * xrz[2] * ps2
        zv1 = zv1 - 3 * xrz[0] * ps2 * ps2 + 4 * xrz[3] * ps1 - 4 * xrz[1] * ps3 - 4 * xrz[0] * ps1 * ps3
        zv1 = zv1 + xrz[4] + j * xrz[0] * ps4
        zv2 = 4 * yrz[1] * ps1 ** 3 + yrz[0] * ps1 ** 4 + 6 * yrz[2] * ps1 ** 2
        zv2 = zv2 + 12j * yrz[1] * ps1 * ps2 + 6j * yrz[0] * ps1 ** 2 * ps2 + 6j * yrz[2] * ps2
        zv2 = zv2 - 3 * yrz[0] * ps2 * ps2 + 4 * yrz[3] * ps1 - 4 * yrz[1] * ps3 - 4 * yrz[0] * ps1 * ps3
        zv2 = zv2 + yrz[4] - j * yrz[0] * ps4
        zv = exptheta * zv1 + zv2 / exptheta
    ctx.prec = wpinitial
    return zv