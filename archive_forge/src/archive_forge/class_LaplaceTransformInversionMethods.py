class LaplaceTransformInversionMethods(object):

    def __init__(ctx, *args, **kwargs):
        ctx._fixed_talbot = FixedTalbot(ctx)
        ctx._stehfest = Stehfest(ctx)
        ctx._de_hoog = deHoog(ctx)
        ctx._cohen = Cohen(ctx)

    def invertlaplace(ctx, f, t, **kwargs):
        """Computes the numerical inverse Laplace transform for a
        Laplace-space function at a given time.  The function being
        evaluated is assumed to be a real-valued function of time.

        The user must supply a Laplace-space function `\\bar{f}(p)`,
        and a desired time at which to estimate the time-domain
        solution `f(t)`.

        A few basic examples of Laplace-space functions with known
        inverses (see references [1,2]) :

        .. math ::

            \\mathcal{L}\\left\\lbrace f(t) \\right\\rbrace=\\bar{f}(p)

        .. math ::

            \\mathcal{L}^{-1}\\left\\lbrace \\bar{f}(p) \\right\\rbrace = f(t)

        .. math ::

            \\bar{f}(p) = \\frac{1}{(p+1)^2}

        .. math ::

            f(t) = t e^{-t}

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> tt = [0.001, 0.01, 0.1, 1, 10]
        >>> fp = lambda p: 1/(p+1)**2
        >>> ft = lambda t: t*exp(-t)
        >>> ft(tt[0]),ft(tt[0])-invertlaplace(fp,tt[0],method='talbot')
        (0.000999000499833375, 8.57923043561212e-20)
        >>> ft(tt[1]),ft(tt[1])-invertlaplace(fp,tt[1],method='talbot')
        (0.00990049833749168, 3.27007646698047e-19)
        >>> ft(tt[2]),ft(tt[2])-invertlaplace(fp,tt[2],method='talbot')
        (0.090483741803596, -1.75215800052168e-18)
        >>> ft(tt[3]),ft(tt[3])-invertlaplace(fp,tt[3],method='talbot')
        (0.367879441171442, 1.2428864009344e-17)
        >>> ft(tt[4]),ft(tt[4])-invertlaplace(fp,tt[4],method='talbot')
        (0.000453999297624849, 4.04513489306658e-20)

        The methods also work for higher precision:

        >>> mp.dps = 100; mp.pretty = True
        >>> nstr(ft(tt[0]),15),nstr(ft(tt[0])-invertlaplace(fp,tt[0],method='talbot'),15)
        ('0.000999000499833375', '-4.96868310693356e-105')
        >>> nstr(ft(tt[1]),15),nstr(ft(tt[1])-invertlaplace(fp,tt[1],method='talbot'),15)
        ('0.00990049833749168', '1.23032291513122e-104')

        .. math ::

            \\bar{f}(p) = \\frac{1}{p^2+1}

        .. math ::

            f(t) = \\mathrm{J}_0(t)

        >>> mp.dps = 15; mp.pretty = True
        >>> fp = lambda p: 1/sqrt(p*p + 1)
        >>> ft = lambda t: besselj(0,t)
        >>> ft(tt[0]),ft(tt[0])-invertlaplace(fp,tt[0],method='dehoog')
        (0.999999750000016, -6.09717765032273e-18)
        >>> ft(tt[1]),ft(tt[1])-invertlaplace(fp,tt[1],method='dehoog')
        (0.99997500015625, -5.61756281076169e-17)

        .. math ::

            \\bar{f}(p) = \\frac{\\log p}{p}

        .. math ::

            f(t) = -\\gamma -\\log t

        >>> mp.dps = 15; mp.pretty = True
        >>> fp = lambda p: log(p)/p
        >>> ft = lambda t: -euler-log(t)
        >>> ft(tt[0]),ft(tt[0])-invertlaplace(fp,tt[0],method='stehfest')
        (6.3305396140806, -1.92126634837863e-16)
        >>> ft(tt[1]),ft(tt[1])-invertlaplace(fp,tt[1],method='stehfest')
        (4.02795452108656, -4.81486093200704e-16)

        **Options**

        :func:`~mpmath.invertlaplace` recognizes the following optional
        keywords valid for all methods:

        *method*
            Chooses numerical inverse Laplace transform algorithm
            (described below).
        *degree*
            Number of terms used in the approximation

        **Algorithms**

        Mpmath implements four numerical inverse Laplace transform
        algorithms, attributed to: Talbot, Stehfest, and de Hoog,
        Knight and Stokes. These can be selected by using
        *method='talbot'*, *method='stehfest'*, *method='dehoog'* or
        *method='cohen'* or by passing the classes *method=FixedTalbot*,
        *method=Stehfest*, *method=deHoog*, or *method=Cohen*. The functions
        :func:`~mpmath.invlaptalbot`, :func:`~mpmath.invlapstehfest`,
        :func:`~mpmath.invlapdehoog`, and :func:`~mpmath.invlapcohen`
        are also available as shortcuts.

        All four algorithms implement a heuristic balance between the
        requested precision and the precision used internally for the
        calculations. This has been tuned for a typical exponentially
        decaying function and precision up to few hundred decimal
        digits.

        The Laplace transform converts the variable time (i.e., along
        a line) into a parameter given by the right half of the
        complex `p`-plane.  Singularities, poles, and branch cuts in
        the complex `p`-plane contain all the information regarding
        the time behavior of the corresponding function. Any numerical
        method must therefore sample `p`-plane "close enough" to the
        singularities to accurately characterize them, while not
        getting too close to have catastrophic cancellation, overflow,
        or underflow issues. Most significantly, if one or more of the
        singularities in the `p`-plane is not on the left side of the
        Bromwich contour, its effects will be left out of the computed
        solution, and the answer will be completely wrong.

        *Talbot*

        The fixed Talbot method is high accuracy and fast, but the
        method can catastrophically fail for certain classes of time-domain
        behavior, including a Heaviside step function for positive
        time (e.g., `H(t-2)`), or some oscillatory behaviors. The
        Talbot method usually has adjustable parameters, but the
        "fixed" variety implemented here does not. This method
        deforms the Bromwich integral contour in the shape of a
        parabola towards `-\\infty`, which leads to problems
        when the solution has a decaying exponential in it (e.g., a
        Heaviside step function is equivalent to multiplying by a
        decaying exponential in Laplace space).

        *Stehfest*

        The Stehfest algorithm only uses abscissa along the real axis
        of the complex `p`-plane to estimate the time-domain
        function. Oscillatory time-domain functions have poles away
        from the real axis, so this method does not work well with
        oscillatory functions, especially high-frequency ones. This
        method also depends on summation of terms in a series that
        grows very large, and will have catastrophic cancellation
        during summation if the working precision is too low.

        *de Hoog et al.*

        The de Hoog, Knight, and Stokes method is essentially a
        Fourier-series quadrature-type approximation to the Bromwich
        contour integral, with non-linear series acceleration and an
        analytical expression for the remainder term. This method is
        typically one of the most robust. This method also involves the
        greatest amount of overhead, so it is typically the slowest of the
        four methods at high precision.

        *Cohen*

        The Cohen method is a trapezoidal rule approximation to the Bromwich
        contour integral, with linear acceleration for alternating
        series. This method is as robust as the de Hoog et al method and the
        fastest of the four methods at high precision, and is therefore the
        default method.

        **Singularities**

        All numerical inverse Laplace transform methods have problems
        at large time when the Laplace-space function has poles,
        singularities, or branch cuts to the right of the origin in
        the complex plane. For simple poles in `\\bar{f}(p)` at the
        `p`-plane origin, the time function is constant in time (e.g.,
        `\\mathcal{L}\\left\\lbrace 1 \\right\\rbrace=1/p` has a pole at
        `p=0`). A pole in `\\bar{f}(p)` to the left of the origin is a
        decreasing function of time (e.g., `\\mathcal{L}\\left\\lbrace
        e^{-t/2} \\right\\rbrace=1/(p+1/2)` has a pole at `p=-1/2`), and
        a pole to the right of the origin leads to an increasing
        function in time (e.g., `\\mathcal{L}\\left\\lbrace t e^{t/4}
        \\right\\rbrace = 1/(p-1/4)^2` has a pole at `p=1/4`).  When
        singularities occur off the real `p` axis, the time-domain
        function is oscillatory. For example `\\mathcal{L}\\left\\lbrace
        \\mathrm{J}_0(t) \\right\\rbrace=1/\\sqrt{p^2+1}` has a branch cut
        starting at `p=j=\\sqrt{-1}` and is a decaying oscillatory
        function, This range of behaviors is illustrated in Duffy [3]
        Figure 4.10.4, p. 228.

        In general as `p \\rightarrow \\infty` `t \\rightarrow 0` and
        vice-versa. All numerical inverse Laplace transform methods
        require their abscissa to shift closer to the origin for
        larger times. If the abscissa shift left of the rightmost
        singularity in the Laplace domain, the answer will be
        completely wrong (the effect of singularities to the right of
        the Bromwich contour are not included in the results).

        For example, the following exponentially growing function has
        a pole at `p=3`:

        .. math ::

            \\bar{f}(p)=\\frac{1}{p^2-9}

        .. math ::

            f(t)=\\frac{1}{3}\\sinh 3t

        >>> mp.dps = 15; mp.pretty = True
        >>> fp = lambda p: 1/(p*p-9)
        >>> ft = lambda t: sinh(3*t)/3
        >>> tt = [0.01,0.1,1.0,10.0]
        >>> ft(tt[0]),invertlaplace(fp,tt[0],method='talbot')
        (0.0100015000675014, 0.0100015000675014)
        >>> ft(tt[1]),invertlaplace(fp,tt[1],method='talbot')
        (0.101506764482381, 0.101506764482381)
        >>> ft(tt[2]),invertlaplace(fp,tt[2],method='talbot')
        (3.33929164246997, 3.33929164246997)
        >>> ft(tt[3]),invertlaplace(fp,tt[3],method='talbot')
        (1781079096920.74, -1.61331069624091e-14)

        **References**

        1. [DLMF]_ section 1.14 (http://dlmf.nist.gov/1.14T4)
        2. Cohen, A.M. (2007). Numerical Methods for Laplace Transform
           Inversion, Springer.
        3. Duffy, D.G. (1998). Advanced Engineering Mathematics, CRC Press.

        **Numerical Inverse Laplace Transform Reviews**

        1. Bellman, R., R.E. Kalaba, J.A. Lockett (1966). *Numerical
           inversion of the Laplace transform: Applications to Biology,
           Economics, Engineering, and Physics*. Elsevier.
        2. Davies, B., B. Martin (1979). Numerical inversion of the
           Laplace transform: a survey and comparison of methods. *Journal
           of Computational Physics* 33:1-32,
           http://dx.doi.org/10.1016/0021-9991(79)90025-1
        3. Duffy, D.G. (1993). On the numerical inversion of Laplace
           transforms: Comparison of three new methods on characteristic
           problems from applications. *ACM Transactions on Mathematical
           Software* 19(3):333-359, http://dx.doi.org/10.1145/155743.155788
        4. Kuhlman, K.L., (2013). Review of Inverse Laplace Transform
           Algorithms for Laplace-Space Numerical Approaches, *Numerical
           Algorithms*, 63(2):339-355.
           http://dx.doi.org/10.1007/s11075-012-9625-3

        """
        rule = kwargs.get('method', 'cohen')
        if type(rule) is str:
            lrule = rule.lower()
            if lrule == 'talbot':
                rule = ctx._fixed_talbot
            elif lrule == 'stehfest':
                rule = ctx._stehfest
            elif lrule == 'dehoog':
                rule = ctx._de_hoog
            elif rule == 'cohen':
                rule = ctx._cohen
            else:
                raise ValueError('unknown invlap algorithm: %s' % rule)
        else:
            rule = rule(ctx)
        rule.calc_laplace_parameter(t, **kwargs)
        fp = [f(p) for p in rule.p]
        return rule.calc_time_domain_solution(fp, t)

    def invlaptalbot(ctx, *args, **kwargs):
        kwargs['method'] = 'talbot'
        return ctx.invertlaplace(*args, **kwargs)

    def invlapstehfest(ctx, *args, **kwargs):
        kwargs['method'] = 'stehfest'
        return ctx.invertlaplace(*args, **kwargs)

    def invlapdehoog(ctx, *args, **kwargs):
        kwargs['method'] = 'dehoog'
        return ctx.invertlaplace(*args, **kwargs)

    def invlapcohen(ctx, *args, **kwargs):
        kwargs['method'] = 'cohen'
        return ctx.invertlaplace(*args, **kwargs)