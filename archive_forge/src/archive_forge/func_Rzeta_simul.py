import math
from .functions import defun
def Rzeta_simul(ctx, s, der=0):
    wpinitial = ctx.prec
    t = ctx._im(s)
    xsigma = ctx._re(s)
    ysigma = 1 - xsigma
    ctx.prec = 15
    a = ctx.sqrt(t / (2 * ctx.pi))
    xasigma = a ** xsigma
    yasigma = a ** ysigma
    xA1 = ctx.power(2, ctx.mag(xasigma) - 1)
    yA1 = ctx.power(2, ctx.mag(yasigma) - 1)
    eps = ctx.power(2, -wpinitial)
    eps1 = eps / 6.0
    xeps2 = eps * xA1 / 3.0
    yeps2 = eps * yA1 / 3.0
    ctx.prec = 15
    if xsigma > 0:
        xb = 2.0
        xc = math.pow(9, xsigma) / 4.44288
        xA = math.pow(9, xsigma)
        xB1 = 1
    else:
        xb = 2.25158
        xc = math.pow(2, -xsigma) / 4.44288
        xA = math.pow(2, -xsigma)
        xB1 = 1.10789
    if ysigma > 0:
        yb = 2.0
        yc = math.pow(9, ysigma) / 4.44288
        yA = math.pow(9, ysigma)
        yB1 = 1
    else:
        yb = 2.25158
        yc = math.pow(2, -ysigma) / 4.44288
        yA = math.pow(2, -ysigma)
        yB1 = 1.10789
    ctx.prec = 15
    xL = 1
    while 3 * xc * ctx.gamma(xL * 0.5) * ctx.power(xb * a, -xL) >= xeps2:
        xL = xL + 1
    xL = max(2, xL)
    yL = 1
    while 3 * yc * ctx.gamma(yL * 0.5) * ctx.power(yb * a, -yL) >= yeps2:
        yL = yL + 1
    yL = max(2, yL)
    if 3 * xL >= 2 * a * a / 25.0 or 3 * xL + 2 + xsigma < 0 or abs(xsigma) > a / 2.0 or (3 * yL >= 2 * a * a / 25.0) or (3 * yL + 2 + ysigma < 0) or (abs(ysigma) > a / 2.0):
        ctx.prec = wpinitial
        raise NotImplementedError('Riemann-Siegel can not compute with such precision')
    L = max(xL, yL)
    xeps3 = xeps2 / (4 * xL)
    yeps3 = yeps2 / (4 * yL)
    xeps4 = xeps3 / (3 * xL)
    yeps4 = yeps3 / (3 * yL)
    xM = aux_M_Fp(ctx, xA, xeps4, a, xB1, xL)
    yM = aux_M_Fp(ctx, yA, yeps4, a, yB1, yL)
    M = max(xM, yM)
    h3 = aux_J_needed(ctx, xA, xeps4, a, xB1, xM)
    h4 = aux_J_needed(ctx, yA, yeps4, a, yB1, yM)
    h3 = min(h3, h4)
    J = 12
    jvalue = (2 * ctx.pi) ** J / ctx.gamma(J + 1)
    while jvalue > h3:
        J = J + 1
        jvalue = 2 * ctx.pi * jvalue / J
    eps5 = {}
    xforeps5 = math.pi * math.pi * xB1 * a
    yforeps5 = math.pi * math.pi * yB1 * a
    for m in range(0, 22):
        xaux1 = math.pow(xforeps5, m / 3) / (316.0 * xA)
        yaux1 = math.pow(yforeps5, m / 3) / (316.0 * yA)
        aux1 = min(xaux1, yaux1)
        aux2 = ctx.gamma(m + 1) / ctx.gamma(m / 3.0 + 0.5)
        aux2 = math.sqrt(aux2)
        eps5[m] = aux1 * aux2 * min(xeps4, yeps4)
    twenty = min(3 * L - 3, 21) + 1
    aux = 6812 * J
    wpfp = ctx.mag(44 * J)
    for m in range(0, twenty):
        wpfp = max(wpfp, ctx.mag(aux * ctx.gamma(m + 1) / eps5[m]))
    ctx.prec = wpfp + ctx.mag(t) + 20
    a = ctx.sqrt(t / (2 * ctx.pi))
    N = ctx.floor(a)
    p = 1 - 2 * (a - N)
    num = ctx.floor(p * ctx.mpf('2') ** wpfp)
    difference = p * ctx.mpf('2') ** wpfp - num
    if difference < 0.5:
        num = num
    else:
        num = num + 1
    p = ctx.convert(num * ctx.mpf('2') ** (-wpfp))
    eps6 = ctx.power(ctx.convert(2 * ctx.pi), J) / (ctx.gamma(J + 1) * 3 * J)
    cc = {}
    cont = {}
    cont, pipowers = coef(ctx, J, eps6)
    cc = cont.copy()
    Fp = {}
    for n in range(M, 3 * L - 2):
        Fp[n] = 0
    Fp = {}
    ctx.prec = wpfp
    for m in range(0, M + 1):
        sumP = 0
        for k in range(2 * J - m - 1, -1, -1):
            sumP = sumP * p + cc[k]
        Fp[m] = sumP
        for k in range(0, 2 * J - m - 1):
            cc[k] = (k + 1) * cc[k + 1]
    xwpd = {}
    d1 = max(6, ctx.mag(40 * L * L))
    xd2 = 13 + ctx.mag((1 + abs(xsigma)) * xA) - ctx.mag(xeps4) - 1
    xconst = ctx.ln(8 / (ctx.pi * ctx.pi * a * a * xB1 * xB1)) / 2
    for n in range(0, L):
        xd3 = ctx.mag(ctx.sqrt(ctx.gamma(n - 0.5))) - ctx.floor(n * xconst) + xd2
        xwpd[n] = max(xd3, d1)
    ctx.prec = xwpd[1] + 10
    xpsigma = 1 - 2 * xsigma
    xd = {}
    xd[0, 0, -2] = 0
    xd[0, 0, -1] = 0
    xd[0, 0, 0] = 1
    xd[0, 0, 1] = 0
    xd[0, -1, -2] = 0
    xd[0, -1, -1] = 0
    xd[0, -1, 0] = 1
    xd[0, -1, 1] = 0
    for n in range(1, L):
        ctx.prec = xwpd[n] + 10
        for k in range(0, 3 * n // 2 + 1):
            m = 3 * n - 2 * k
            if m != 0:
                m1 = ctx.one / m
                c1 = m1 / 4
                c2 = xpsigma * m1 / 2
                c3 = -(m + 1)
                xd[0, n, k] = c3 * xd[0, n - 1, k - 2] + c1 * xd[0, n - 1, k] + c2 * xd[0, n - 1, k - 1]
            else:
                xd[0, n, k] = 0
                for r in range(0, k):
                    add = xd[0, n, r] * (ctx.mpf('1.0') * ctx.fac(2 * k - 2 * r) / ctx.fac(k - r))
                    xd[0, n, k] -= (-1) ** (k - r) * add
        xd[0, n, -2] = 0
        xd[0, n, -1] = 0
        xd[0, n, 3 * n // 2 + 1] = 0
    for mu in range(-2, der + 1):
        for n in range(-2, L):
            for k in range(-3, max(1, 3 * n // 2 + 2)):
                if mu < 0 or n < 0 or k < 0 or (k > 3 * n // 2):
                    xd[mu, n, k] = 0
    for mu in range(1, der + 1):
        for n in range(0, L):
            ctx.prec = xwpd[n] + 10
            for k in range(0, 3 * n // 2 + 1):
                aux = (2 * mu - 2) * xd[mu - 2, n - 2, k - 3] + 2 * (xsigma + n - 2) * xd[mu - 1, n - 2, k - 3]
                xd[mu, n, k] = aux - xd[mu - 1, n - 1, k - 1]
    ywpd = {}
    d1 = max(6, ctx.mag(40 * L * L))
    yd2 = 13 + ctx.mag((1 + abs(ysigma)) * yA) - ctx.mag(yeps4) - 1
    yconst = ctx.ln(8 / (ctx.pi * ctx.pi * a * a * yB1 * yB1)) / 2
    for n in range(0, L):
        yd3 = ctx.mag(ctx.sqrt(ctx.gamma(n - 0.5))) - ctx.floor(n * yconst) + yd2
        ywpd[n] = max(yd3, d1)
    ctx.prec = ywpd[1] + 10
    ypsigma = 1 - 2 * ysigma
    yd = {}
    yd[0, 0, -2] = 0
    yd[0, 0, -1] = 0
    yd[0, 0, 0] = 1
    yd[0, 0, 1] = 0
    yd[0, -1, -2] = 0
    yd[0, -1, -1] = 0
    yd[0, -1, 0] = 1
    yd[0, -1, 1] = 0
    for n in range(1, L):
        ctx.prec = ywpd[n] + 10
        for k in range(0, 3 * n // 2 + 1):
            m = 3 * n - 2 * k
            if m != 0:
                m1 = ctx.one / m
                c1 = m1 / 4
                c2 = ypsigma * m1 / 2
                c3 = -(m + 1)
                yd[0, n, k] = c3 * yd[0, n - 1, k - 2] + c1 * yd[0, n - 1, k] + c2 * yd[0, n - 1, k - 1]
            else:
                yd[0, n, k] = 0
                for r in range(0, k):
                    add = yd[0, n, r] * (ctx.mpf('1.0') * ctx.fac(2 * k - 2 * r) / ctx.fac(k - r))
                    yd[0, n, k] -= (-1) ** (k - r) * add
        yd[0, n, -2] = 0
        yd[0, n, -1] = 0
        yd[0, n, 3 * n // 2 + 1] = 0
    for mu in range(-2, der + 1):
        for n in range(-2, L):
            for k in range(-3, max(1, 3 * n // 2 + 2)):
                if mu < 0 or n < 0 or k < 0 or (k > 3 * n // 2):
                    yd[mu, n, k] = 0
    for mu in range(1, der + 1):
        for n in range(0, L):
            ctx.prec = ywpd[n] + 10
            for k in range(0, 3 * n // 2 + 1):
                aux = (2 * mu - 2) * yd[mu - 2, n - 2, k - 3] + 2 * (ysigma + n - 2) * yd[mu - 1, n - 2, k - 3]
                yd[mu, n, k] = aux - yd[mu - 1, n - 1, k - 1]
    xwptcoef = {}
    xwpterm = {}
    ctx.prec = 15
    c1 = ctx.mag(40 * (L + 2))
    xc2 = ctx.mag(68 * (L + 2) * xA)
    xc4 = ctx.mag(xB1 * a * math.sqrt(ctx.pi)) - 1
    for k in range(0, L):
        xc3 = xc2 - k * xc4 + ctx.mag(ctx.fac(k + 0.5)) / 2.0
        xwptcoef[k] = (max(c1, xc3 - ctx.mag(xeps4) + 1) + 1 + 20) * 1.5
        xwpterm[k] = max(c1, ctx.mag(L + 2) + xc3 - ctx.mag(xeps3) + 1) + 1 + 20
    ywptcoef = {}
    ywpterm = {}
    ctx.prec = 15
    c1 = ctx.mag(40 * (L + 2))
    yc2 = ctx.mag(68 * (L + 2) * yA)
    yc4 = ctx.mag(yB1 * a * math.sqrt(ctx.pi)) - 1
    for k in range(0, L):
        yc3 = yc2 - k * yc4 + ctx.mag(ctx.fac(k + 0.5)) / 2.0
        ywptcoef[k] = (max(c1, yc3 - ctx.mag(yeps4) + 1) + 10) * 1.5
        ywpterm[k] = max(c1, ctx.mag(L + 2) + yc3 - ctx.mag(yeps3) + 1) + 1 + 10
    xfortcoef = {}
    for mu in range(0, der + 1):
        for k in range(0, L):
            for ell in range(-2, 3 * k // 2 + 1):
                xfortcoef[mu, k, ell] = 0
    for mu in range(0, der + 1):
        for k in range(0, L):
            ctx.prec = xwptcoef[k]
            for ell in range(0, 3 * k // 2 + 1):
                xfortcoef[mu, k, ell] = xd[mu, k, ell] * Fp[3 * k - 2 * ell] / pipowers[2 * k - ell]
                xfortcoef[mu, k, ell] = xfortcoef[mu, k, ell] / (2 * ctx.j) ** ell

    def trunc_a(t):
        wp = ctx.prec
        ctx.prec = wp + 2
        aa = ctx.sqrt(t / (2 * ctx.pi))
        ctx.prec = wp
        return aa
    xtcoef = {}
    for mu in range(0, der + 1):
        for k in range(0, L):
            for ell in range(-2, 3 * k // 2 + 1):
                xtcoef[mu, k, ell] = 0
    ctx.prec = max(xwptcoef[0], ywptcoef[0]) + 3
    aa = trunc_a(t)
    la = -ctx.ln(aa)
    for chi in range(0, der + 1):
        for k in range(0, L):
            ctx.prec = xwptcoef[k]
            for ell in range(0, 3 * k // 2 + 1):
                xtcoef[chi, k, ell] = 0
                for mu in range(0, chi + 1):
                    tcoefter = ctx.binomial(chi, mu) * ctx.power(la, mu) * xfortcoef[chi - mu, k, ell]
                    xtcoef[chi, k, ell] += tcoefter
    yfortcoef = {}
    for mu in range(0, der + 1):
        for k in range(0, L):
            for ell in range(-2, 3 * k // 2 + 1):
                yfortcoef[mu, k, ell] = 0
    for mu in range(0, der + 1):
        for k in range(0, L):
            ctx.prec = ywptcoef[k]
            for ell in range(0, 3 * k // 2 + 1):
                yfortcoef[mu, k, ell] = yd[mu, k, ell] * Fp[3 * k - 2 * ell] / pipowers[2 * k - ell]
                yfortcoef[mu, k, ell] = yfortcoef[mu, k, ell] / (2 * ctx.j) ** ell
    ytcoef = {}
    for chi in range(0, der + 1):
        for k in range(0, L):
            for ell in range(-2, 3 * k // 2 + 1):
                ytcoef[chi, k, ell] = 0
    for chi in range(0, der + 1):
        for k in range(0, L):
            ctx.prec = ywptcoef[k]
            for ell in range(0, 3 * k // 2 + 1):
                ytcoef[chi, k, ell] = 0
                for mu in range(0, chi + 1):
                    tcoefter = ctx.binomial(chi, mu) * ctx.power(la, mu) * yfortcoef[chi - mu, k, ell]
                    ytcoef[chi, k, ell] += tcoefter
    ctx.prec = max(xwptcoef[0], ywptcoef[0]) + 2
    av = {}
    av[0] = 1
    av[1] = av[0] / a
    ctx.prec = max(xwptcoef[0], ywptcoef[0])
    for k in range(2, L):
        av[k] = av[k - 1] * av[1]
    xtv = {}
    for chi in range(0, der + 1):
        for k in range(0, L):
            ctx.prec = xwptcoef[k]
            for ell in range(0, 3 * k // 2 + 1):
                xtv[chi, k, ell] = xtcoef[chi, k, ell] * av[k]
    ytv = {}
    for chi in range(0, der + 1):
        for k in range(0, L):
            ctx.prec = ywptcoef[k]
            for ell in range(0, 3 * k // 2 + 1):
                ytv[chi, k, ell] = ytcoef[chi, k, ell] * av[k]
    xterm = {}
    for chi in range(0, der + 1):
        for n in range(0, L):
            ctx.prec = xwpterm[n]
            te = 0
            for k in range(0, 3 * n // 2 + 1):
                te += xtv[chi, n, k]
            xterm[chi, n] = te
    yterm = {}
    for chi in range(0, der + 1):
        for n in range(0, L):
            ctx.prec = ywpterm[n]
            te = 0
            for k in range(0, 3 * n // 2 + 1):
                te += ytv[chi, n, k]
            yterm[chi, n] = te
    xrssum = {}
    ctx.prec = 15
    xrsbound = math.sqrt(ctx.pi) * xc / (xb * a)
    ctx.prec = 15
    xwprssum = ctx.mag(4.4 * (L + 3) ** 2 * xrsbound / xeps2)
    xwprssum = max(xwprssum, ctx.mag(10 * (L + 1)))
    ctx.prec = xwprssum
    for chi in range(0, der + 1):
        xrssum[chi] = 0
        for k in range(1, L + 1):
            xrssum[chi] += xterm[chi, L - k]
    yrssum = {}
    ctx.prec = 15
    yrsbound = math.sqrt(ctx.pi) * yc / (yb * a)
    ctx.prec = 15
    ywprssum = ctx.mag(4.4 * (L + 3) ** 2 * yrsbound / yeps2)
    ywprssum = max(ywprssum, ctx.mag(10 * (L + 1)))
    ctx.prec = ywprssum
    for chi in range(0, der + 1):
        yrssum[chi] = 0
        for k in range(1, L + 1):
            yrssum[chi] += yterm[chi, L - k]
    ctx.prec = 15
    A2 = 2 ** max(ctx.mag(abs(xrssum[0])), ctx.mag(abs(yrssum[0])))
    eps8 = eps / (3 * A2)
    T = t * ctx.ln(t / (2 * ctx.pi))
    xwps3 = 5 + ctx.mag((1 + 2 / eps8 * ctx.power(a, -xsigma)) * T)
    ywps3 = 5 + ctx.mag((1 + 2 / eps8 * ctx.power(a, -ysigma)) * T)
    ctx.prec = max(xwps3, ywps3)
    tpi = t / (2 * ctx.pi)
    arg = t / 2 * ctx.ln(tpi) - t / 2 - ctx.pi / 8
    U = ctx.expj(-arg)
    a = trunc_a(t)
    xasigma = ctx.power(a, -xsigma)
    yasigma = ctx.power(a, -ysigma)
    xS3 = (-1) ** (N - 1) * xasigma * U
    yS3 = (-1) ** (N - 1) * yasigma * U
    ctx.prec = 15
    xwpsum = 4 + ctx.mag((N + ctx.power(N, 1 - xsigma)) * ctx.ln(N) / eps1)
    ywpsum = 4 + ctx.mag((N + ctx.power(N, 1 - ysigma)) * ctx.ln(N) / eps1)
    wpsum = max(xwpsum, ywpsum)
    ctx.prec = wpsum + 10
    '\n    # This can be improved\n    xS1={}\n    yS1={}\n    for chi in range(0,der+1):\n        xS1[chi] = 0\n        yS1[chi] = 0\n    for n in range(1,int(N)+1):\n        ln = ctx.ln(n)\n        xexpn = ctx.exp(-ln*(xsigma+ctx.j*t))\n        yexpn = ctx.conj(1/(n*xexpn))\n        for chi in range(0,der+1):\n            pown = ctx.power(-ln, chi)\n            xterm = pown*xexpn\n            yterm = pown*yexpn\n            xS1[chi] += xterm\n            yS1[chi] += yterm\n    '
    xS1, yS1 = ctx._zetasum(s, 1, int(N) - 1, range(0, der + 1), True)
    ctx.prec = 15
    xabsS1 = abs(xS1[der])
    xabsS2 = abs(xrssum[der] * xS3)
    xwpend = max(6, wpinitial + ctx.mag(6 * (3 * xabsS1 + 7 * xabsS2)))
    ctx.prec = xwpend
    xrz = {}
    for chi in range(0, der + 1):
        xrz[chi] = xS1[chi] + xrssum[chi] * xS3
    ctx.prec = 15
    yabsS1 = abs(yS1[der])
    yabsS2 = abs(yrssum[der] * yS3)
    ywpend = max(6, wpinitial + ctx.mag(6 * (3 * yabsS1 + 7 * yabsS2)))
    ctx.prec = ywpend
    yrz = {}
    for chi in range(0, der + 1):
        yrz[chi] = yS1[chi] + yrssum[chi] * yS3
        yrz[chi] = ctx.conj(yrz[chi])
    ctx.prec = wpinitial
    return (xrz, yrz)