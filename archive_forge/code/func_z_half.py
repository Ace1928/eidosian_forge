import math
from .functions import defun
def z_half(ctx, t, der=0):
    """
    z_half(t,der=0) Computes Z^(der)(t)
    """
    s = ctx.mpf('0.5') + ctx.j * t
    wpinitial = ctx.prec
    ctx.prec = 15
    tt = t / (2 * ctx.pi)
    wptheta = wpinitial + 1 + ctx.mag(3 * tt ** 1.5 * ctx.ln(tt))
    wpz = wpinitial + 1 + ctx.mag(12 * tt * ctx.ln(tt))
    ctx.prec = wptheta
    theta = ctx.siegeltheta(t)
    ctx.prec = wpz
    rz = Rzeta_set(ctx, s, range(der + 1))
    if der > 0:
        ps1 = ctx._re(ctx.psi(0, s / 2) / 2 - ctx.ln(ctx.pi) / 2)
    if der > 1:
        ps2 = ctx._re(ctx.j * ctx.psi(1, s / 2) / 4)
    if der > 2:
        ps3 = ctx._re(-ctx.psi(2, s / 2) / 8)
    if der > 3:
        ps4 = ctx._re(-ctx.j * ctx.psi(3, s / 2) / 16)
    exptheta = ctx.expj(theta)
    if der == 0:
        z = 2 * exptheta * rz[0]
    if der == 1:
        zf = 2j * exptheta
        z = zf * (ps1 * rz[0] + rz[1])
    if der == 2:
        zf = 2 * exptheta
        z = -zf * (2 * rz[1] * ps1 + rz[0] * ps1 ** 2 + rz[2] - ctx.j * rz[0] * ps2)
    if der == 3:
        zf = -2j * exptheta
        z = 3 * rz[1] * ps1 ** 2 + rz[0] * ps1 ** 3 + 3 * ps1 * rz[2]
        z = zf * (z - 3j * rz[1] * ps2 - 3j * rz[0] * ps1 * ps2 + rz[3] - rz[0] * ps3)
    if der == 4:
        zf = 2 * exptheta
        z = 4 * rz[1] * ps1 ** 3 + rz[0] * ps1 ** 4 + 6 * ps1 ** 2 * rz[2]
        z = z - 12j * rz[1] * ps1 * ps2 - 6j * rz[0] * ps1 ** 2 * ps2 - 6j * rz[2] * ps2 - 3 * rz[0] * ps2 * ps2
        z = z + 4 * ps1 * rz[3] - 4 * rz[1] * ps3 - 4 * rz[0] * ps1 * ps3 + rz[4] + ctx.j * rz[0] * ps4
        z = zf * z
    ctx.prec = wpinitial
    return ctx._re(z)