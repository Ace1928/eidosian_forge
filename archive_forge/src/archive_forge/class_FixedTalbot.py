class FixedTalbot(InverseLaplaceTransform):

    def calc_laplace_parameter(self, t, **kwargs):
        """The "fixed" Talbot method deforms the Bromwich contour towards
        `-\\infty` in the shape of a parabola. Traditionally the Talbot
        algorithm has adjustable parameters, but the "fixed" version
        does not. The `r` parameter could be passed in as a parameter,
        if you want to override the default given by (Abate & Valko,
        2004).

        The Laplace parameter is sampled along a parabola opening
        along the negative imaginary axis, with the base of the
        parabola along the real axis at
        `p=\\frac{r}{t_\\mathrm{max}}`. As the number of terms used in
        the approximation (degree) grows, the abscissa required for
        function evaluation tend towards `-\\infty`, requiring high
        precision to prevent overflow.  If any poles, branch cuts or
        other singularities exist such that the deformed Bromwich
        contour lies to the left of the singularity, the method will
        fail.

        **Optional arguments**

        :class:`~mpmath.calculus.inverselaplace.FixedTalbot.calc_laplace_parameter`
        recognizes the following keywords

        *tmax*
            maximum time associated with vector of times
            (typically just the time requested)
        *degree*
            integer order of approximation (M = number of terms)
        *r*
            abscissa for `p_0` (otherwise computed using rule
            of thumb `2M/5`)

        The working precision will be increased according to a rule of
        thumb. If 'degree' is not specified, the working precision and
        degree are chosen to hopefully achieve the dps of the calling
        context. If 'degree' is specified, the working precision is
        chosen to achieve maximum resulting precision for the
        specified degree.

        .. math ::

            p_0=\\frac{r}{t}

        .. math ::

            p_i=\\frac{i r \\pi}{Mt_\\mathrm{max}}\\left[\\cot\\left(
            \\frac{i\\pi}{M}\\right) + j \\right] \\qquad 1\\le i <M

        where `j=\\sqrt{-1}`, `r=2M/5`, and `t_\\mathrm{max}` is the
        maximum specified time.

        """
        self.t = self.ctx.convert(t)
        self.tmax = self.ctx.convert(kwargs.get('tmax', self.t))
        if 'degree' in kwargs:
            self.degree = kwargs['degree']
            self.dps_goal = self.degree
        else:
            self.dps_goal = int(1.72 * self.ctx.dps)
            self.degree = max(12, int(1.38 * self.dps_goal))
        M = self.degree
        self.dps_orig = self.ctx.dps
        self.ctx.dps = self.dps_goal
        self.r = kwargs.get('r', self.ctx.fraction(2, 5) * M)
        self.theta = self.ctx.linspace(0.0, self.ctx.pi, M + 1)
        self.cot_theta = self.ctx.matrix(M, 1)
        self.cot_theta[0] = 0
        self.delta = self.ctx.matrix(M, 1)
        self.delta[0] = self.r
        for i in range(1, M):
            self.cot_theta[i] = self.ctx.cot(self.theta[i])
            self.delta[i] = self.r * self.theta[i] * (self.cot_theta[i] + 1j)
        self.p = self.ctx.matrix(M, 1)
        self.p = self.delta / self.tmax

    def calc_time_domain_solution(self, fp, t, manual_prec=False):
        """The fixed Talbot time-domain solution is computed from the
        Laplace-space function evaluations using

        .. math ::

            f(t,M)=\\frac{2}{5t}\\sum_{k=0}^{M-1}\\Re \\left[
            \\gamma_k \\bar{f}(p_k)\\right]

        where

        .. math ::

            \\gamma_0 = \\frac{1}{2}e^{r}\\bar{f}(p_0)

        .. math ::

            \\gamma_k = e^{tp_k}\\left\\lbrace 1 + \\frac{jk\\pi}{M}\\left[1 +
            \\cot \\left( \\frac{k \\pi}{M} \\right)^2 \\right] - j\\cot\\left(
            \\frac{k \\pi}{M}\\right)\\right \\rbrace \\qquad 1\\le k<M.

        Again, `j=\\sqrt{-1}`.

        Before calling this function, call
        :class:`~mpmath.calculus.inverselaplace.FixedTalbot.calc_laplace_parameter`
        to set the parameters and compute the required coefficients.

        **References**

        1. Abate, J., P. Valko (2004). Multi-precision Laplace
           transform inversion. *International Journal for Numerical
           Methods in Engineering* 60:979-993,
           http://dx.doi.org/10.1002/nme.995
        2. Talbot, A. (1979). The accurate numerical inversion of
           Laplace transforms. *IMA Journal of Applied Mathematics*
           23(1):97, http://dx.doi.org/10.1093/imamat/23.1.97
        """
        self.t = self.ctx.convert(t)
        theta = self.theta
        delta = self.delta
        M = self.degree
        p = self.p
        r = self.r
        ans = self.ctx.matrix(M, 1)
        ans[0] = self.ctx.exp(delta[0]) * fp[0] / 2
        for i in range(1, M):
            ans[i] = self.ctx.exp(delta[i]) * fp[i] * (1 + 1j * theta[i] * (1 + self.cot_theta[i] ** 2) - 1j * self.cot_theta[i])
        result = self.ctx.fraction(2, 5) * self.ctx.fsum(ans) / self.t
        if not manual_prec:
            self.ctx.dps = self.dps_orig
        return result.real