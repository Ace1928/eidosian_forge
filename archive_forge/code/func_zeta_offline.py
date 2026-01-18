import math
from .functions import defun
def zeta_offline(ctx, s, k=0):
    """
    Computes zeta^(k)(s) off the line
    """
    wpinitial = ctx.prec
    sigma = ctx._re(s)
    t = ctx._im(s)
    ctx.prec = 53
    if sigma > 0:
        X = ctx.power(abs(s), 0.5)
    else:
        X = ctx.power(2 * ctx.pi, sigma - 1) * ctx.power(abs(1 - s), 0.5 - sigma)
    if sigma > 0:
        M1 = 2 * ctx.sqrt(t / (2 * ctx.pi))
    else:
        M1 = 4 * t * X
    if 1 - sigma > 0:
        M2 = 2 * ctx.sqrt(t / (2 * ctx.pi))
    else:
        M2 = 4 * t * ctx.power(2 * ctx.pi, -sigma) * ctx.power(abs(s), sigma - 0.5)
    abst = abs(0.5 - s)
    T = 2 * abst * math.log(abst)
    wpbasic = max(6, 3 + ctx.mag(t))
    wpbasic2 = 2 + ctx.mag(2.12 * M1 + 21.2 * M2 * X + 1.3 * M2 * X * T) + wpinitial + 1
    wpbasic = max(wpbasic, wpbasic2)
    wptheta = max(4, 3 + ctx.mag(2.7 * M2 * X) + wpinitial + 1)
    wpR = 3 + ctx.mag(1.1 + 2 * X) + wpinitial + 1
    ctx.prec = wptheta
    theta = ctx.siegeltheta(t - ctx.j * (sigma - ctx.mpf('0.5')))
    s1 = s
    s2 = ctx.conj(1 - s1)
    ctx.prec = wpR
    xrz, yrz = Rzeta_simul(ctx, s, k)
    if k > 0:
        ps1 = (ctx.psi(0, s1 / 2) + ctx.psi(0, (1 - s1) / 2)) / 4 - ctx.ln(ctx.pi) / 2
    if k > 1:
        ps2 = ctx.j * (ctx.psi(1, s1 / 2) - ctx.psi(1, (1 - s1) / 2)) / 8
    if k > 2:
        ps3 = -(ctx.psi(2, s1 / 2) + ctx.psi(2, (1 - s1) / 2)) / 16
    if k > 3:
        ps4 = -ctx.j * (ctx.psi(3, s1 / 2) - ctx.psi(3, (1 - s1) / 2)) / 32
    ctx.prec = wpbasic
    exptheta = ctx.expj(-2 * theta)
    if k == 0:
        zv = xrz[0] + exptheta * yrz[0]
    if k == 1:
        zv1 = -yrz[1] - 2 * yrz[0] * ps1
        zv = xrz[1] + exptheta * zv1
    if k == 2:
        zv1 = 4 * yrz[1] * ps1 + 4 * yrz[0] * ps1 ** 2 + yrz[2] + 2j * yrz[0] * ps2
        zv = xrz[2] + exptheta * zv1
    if k == 3:
        zv1 = -12 * yrz[1] * ps1 ** 2 - 8 * yrz[0] * ps1 ** 3 - 6 * yrz[2] * ps1 - 6j * yrz[1] * ps2
        zv1 = zv1 - 12j * yrz[0] * ps1 * ps2 - yrz[3] + 2 * yrz[0] * ps3
        zv = xrz[3] + exptheta * zv1
    if k == 4:
        zv1 = 32 * yrz[1] * ps1 ** 3 + 16 * yrz[0] * ps1 ** 4 + 24 * yrz[2] * ps1 ** 2
        zv1 = zv1 + 48j * yrz[1] * ps1 * ps2 + 48j * yrz[0] * ps1 ** 2 * ps2
        zv1 = zv1 + 12j * yrz[2] * ps2 - 12 * yrz[0] * ps2 ** 2 + 8 * yrz[3] * ps1 - 8 * yrz[1] * ps3
        zv1 = zv1 - 16 * yrz[0] * ps1 * ps3 + yrz[4] - 2j * yrz[0] * ps4
        zv = xrz[4] + exptheta * zv1
    ctx.prec = wpinitial
    return zv