from .functions import defun, defun_wrapped
@defun
def _jacobi_theta2(ctx, z, q):
    extra1 = 10
    extra2 = 20
    MIN = 2
    if z == ctx.zero:
        if not ctx._im(q):
            wp = ctx.prec + extra1
            x = ctx.to_fixed(ctx._re(q), wp)
            x2 = x * x >> wp
            a = b = x2
            s = x2
            while abs(a) > MIN:
                b = b * x2 >> wp
                a = a * b >> wp
                s += a
            s = (1 << wp + 1) + (s << 1)
            s = ctx.ldexp(s, -wp)
        else:
            wp = ctx.prec + extra1
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = xre * xre - xim * xim >> wp
            x2im = xre * xim >> wp - 1
            are = bre = x2re
            aim = bim = x2im
            sre = (1 << wp) + are
            sim = aim
            while are ** 2 + aim ** 2 > MIN:
                bre, bim = (bre * x2re - bim * x2im >> wp, bre * x2im + bim * x2re >> wp)
                are, aim = (are * bre - aim * bim >> wp, are * bim + aim * bre >> wp)
                sre += are
                sim += aim
            sre = sre << 1
            sim = sim << 1
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
    elif not ctx._im(q) and (not ctx._im(z)):
        wp = ctx.prec + extra1
        x = ctx.to_fixed(ctx._re(q), wp)
        x2 = x * x >> wp
        a = b = x2
        c1, s1 = ctx.cos_sin(ctx._re(z), prec=wp)
        cn = c1 = ctx.to_fixed(c1, wp)
        sn = s1 = ctx.to_fixed(s1, wp)
        c2 = c1 * c1 - s1 * s1 >> wp
        s2 = c1 * s1 >> wp - 1
        cn, sn = (cn * c2 - sn * s2 >> wp, sn * c2 + cn * s2 >> wp)
        s = c1 + (a * cn >> wp)
        while abs(a) > MIN:
            b = b * x2 >> wp
            a = a * b >> wp
            cn, sn = (cn * c2 - sn * s2 >> wp, sn * c2 + cn * s2 >> wp)
            s += a * cn >> wp
        s = s << 1
        s = ctx.ldexp(s, -wp)
        s *= ctx.nthroot(q, 4)
        return s
    elif not ctx._im(z):
        wp = ctx.prec + extra2
        xre = ctx.to_fixed(ctx._re(q), wp)
        xim = ctx.to_fixed(ctx._im(q), wp)
        x2re = xre * xre - xim * xim >> wp
        x2im = xre * xim >> wp - 1
        are = bre = x2re
        aim = bim = x2im
        c1, s1 = ctx.cos_sin(ctx._re(z), prec=wp)
        cn = c1 = ctx.to_fixed(c1, wp)
        sn = s1 = ctx.to_fixed(s1, wp)
        c2 = c1 * c1 - s1 * s1 >> wp
        s2 = c1 * s1 >> wp - 1
        cn, sn = (cn * c2 - sn * s2 >> wp, sn * c2 + cn * s2 >> wp)
        sre = c1 + (are * cn >> wp)
        sim = aim * cn >> wp
        while are ** 2 + aim ** 2 > MIN:
            bre, bim = (bre * x2re - bim * x2im >> wp, bre * x2im + bim * x2re >> wp)
            are, aim = (are * bre - aim * bim >> wp, are * bim + aim * bre >> wp)
            cn, sn = (cn * c2 - sn * s2 >> wp, sn * c2 + cn * s2 >> wp)
            sre += are * cn >> wp
            sim += aim * cn >> wp
        sre = sre << 1
        sim = sim << 1
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    elif not ctx._im(q):
        wp = ctx.prec + extra2
        x = ctx.to_fixed(ctx._re(q), wp)
        x2 = x * x >> wp
        a = b = x2
        prec0 = ctx.prec
        ctx.prec = wp
        c1, s1 = ctx.cos_sin(z)
        ctx.prec = prec0
        cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
        cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
        snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
        snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
        c2re = c1re * c1re - c1im * c1im - s1re * s1re + s1im * s1im >> wp
        c2im = c1re * c1im - s1re * s1im >> wp - 1
        s2re = c1re * s1re - c1im * s1im >> wp - 1
        s2im = c1re * s1im + c1im * s1re >> wp - 1
        t1 = cnre * c2re - cnim * c2im - snre * s2re + snim * s2im >> wp
        t2 = cnre * c2im + cnim * c2re - snre * s2im - snim * s2re >> wp
        t3 = snre * c2re - snim * c2im + cnre * s2re - cnim * s2im >> wp
        t4 = snre * c2im + snim * c2re + cnre * s2im + cnim * s2re >> wp
        cnre = t1
        cnim = t2
        snre = t3
        snim = t4
        sre = c1re + (a * cnre >> wp)
        sim = c1im + (a * cnim >> wp)
        while abs(a) > MIN:
            b = b * x2 >> wp
            a = a * b >> wp
            t1 = cnre * c2re - cnim * c2im - snre * s2re + snim * s2im >> wp
            t2 = cnre * c2im + cnim * c2re - snre * s2im - snim * s2re >> wp
            t3 = snre * c2re - snim * c2im + cnre * s2re - cnim * s2im >> wp
            t4 = snre * c2im + snim * c2re + cnre * s2im + cnim * s2re >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            sre += a * cnre >> wp
            sim += a * cnim >> wp
        sre = sre << 1
        sim = sim << 1
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    else:
        wp = ctx.prec + extra2
        xre = ctx.to_fixed(ctx._re(q), wp)
        xim = ctx.to_fixed(ctx._im(q), wp)
        x2re = xre * xre - xim * xim >> wp
        x2im = xre * xim >> wp - 1
        are = bre = x2re
        aim = bim = x2im
        prec0 = ctx.prec
        ctx.prec = wp
        c1, s1 = ctx.cos_sin(z)
        ctx.prec = prec0
        cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
        cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
        snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
        snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
        c2re = c1re * c1re - c1im * c1im - s1re * s1re + s1im * s1im >> wp
        c2im = c1re * c1im - s1re * s1im >> wp - 1
        s2re = c1re * s1re - c1im * s1im >> wp - 1
        s2im = c1re * s1im + c1im * s1re >> wp - 1
        t1 = cnre * c2re - cnim * c2im - snre * s2re + snim * s2im >> wp
        t2 = cnre * c2im + cnim * c2re - snre * s2im - snim * s2re >> wp
        t3 = snre * c2re - snim * c2im + cnre * s2re - cnim * s2im >> wp
        t4 = snre * c2im + snim * c2re + cnre * s2im + cnim * s2re >> wp
        cnre = t1
        cnim = t2
        snre = t3
        snim = t4
        n = 1
        termre = c1re
        termim = c1im
        sre = c1re + (are * cnre - aim * cnim >> wp)
        sim = c1im + (are * cnim + aim * cnre >> wp)
        n = 3
        termre = are * cnre - aim * cnim >> wp
        termim = are * cnim + aim * cnre >> wp
        sre = c1re + (are * cnre - aim * cnim >> wp)
        sim = c1im + (are * cnim + aim * cnre >> wp)
        n = 5
        while are ** 2 + aim ** 2 > MIN:
            bre, bim = (bre * x2re - bim * x2im >> wp, bre * x2im + bim * x2re >> wp)
            are, aim = (are * bre - aim * bim >> wp, are * bim + aim * bre >> wp)
            t1 = cnre * c2re - cnim * c2im - snre * s2re + snim * s2im >> wp
            t2 = cnre * c2im + cnim * c2re - snre * s2im - snim * s2re >> wp
            t3 = snre * c2re - snim * c2im + cnre * s2re - cnim * s2im >> wp
            t4 = snre * c2im + snim * c2re + cnre * s2im + cnim * s2re >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            termre = are * cnre - aim * cnim >> wp
            termim = aim * cnre + are * cnim >> wp
            sre += are * cnre - aim * cnim >> wp
            sim += aim * cnre + are * cnim >> wp
            n += 2
        sre = sre << 1
        sim = sim << 1
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    s *= ctx.nthroot(q, 4)
    return s