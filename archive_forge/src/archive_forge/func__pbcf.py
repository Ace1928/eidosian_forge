import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def _pbcf(n, theta):
    """Asymptotic series expansion of parabolic cylinder function

    The implementation is based on sections 3.2 and 3.3 from the
    original paper. Compared to the published version this code
    adds one more term to the asymptotic series. The detailed
    formulas can be found at [parabolic-asymptotics]_. The evaluation
    is done in a transformed variable :math:`\\theta := \\arccos(t)`
    where :math:`t := x / \\mu` and :math:`\\mu := \\sqrt{2n + 1}`.

    Parameters
    ----------
    n : int
        Quadrature order
    theta : ndarray
        Transformed position variable

    Returns
    -------
    U : ndarray
        Value of the parabolic cylinder function :math:`U(a, \\theta)`.
    Ud : ndarray
        Value of the derivative :math:`U^{\\prime}(a, \\theta)` of
        the parabolic cylinder function.

    See Also
    --------
    roots_hermite_asy

    References
    ----------
    .. [parabolic-asymptotics]
       https://dlmf.nist.gov/12.10#vii
    """
    st = sin(theta)
    ct = cos(theta)
    mu = 2.0 * n + 1.0
    eta = 0.5 * theta - 0.5 * st * ct
    zeta = -(3.0 * eta / 2.0) ** (2.0 / 3.0)
    phi = (-zeta / st ** 2) ** 0.25
    a0 = 1.0
    a1 = 0.10416666666666667
    a2 = 0.08355034722222222
    a3 = 0.12822657455632716
    a4 = 0.29184902646414046
    a5 = 0.8816272674437576
    b0 = 1.0
    b1 = -0.14583333333333334
    b2 = -0.09874131944444445
    b3 = -0.14331205391589505
    b4 = -0.31722720267841353
    b5 = -0.9424291479571203
    ctp = ct ** arange(16).reshape((-1, 1))
    u0 = 1.0
    u1 = (1.0 * ctp[3, :] - 6.0 * ct) / 24.0
    u2 = (-9.0 * ctp[4, :] + 249.0 * ctp[2, :] + 145.0) / 1152.0
    u3 = (-4042.0 * ctp[9, :] + 18189.0 * ctp[7, :] - 28287.0 * ctp[5, :] - 151995.0 * ctp[3, :] - 259290.0 * ct) / 414720.0
    u4 = (72756.0 * ctp[10, :] - 321339.0 * ctp[8, :] - 154982.0 * ctp[6, :] + 50938215.0 * ctp[4, :] + 122602962.0 * ctp[2, :] + 12773113.0) / 39813120.0
    u5 = (82393456.0 * ctp[15, :] - 617950920.0 * ctp[13, :] + 1994971575.0 * ctp[11, :] - 3630137104.0 * ctp[9, :] + 4433574213.0 * ctp[7, :] - 37370295816.0 * ctp[5, :] - 119582875013.0 * ctp[3, :] - 34009066266.0 * ct) / 6688604160.0
    v0 = 1.0
    v1 = (1.0 * ctp[3, :] + 6.0 * ct) / 24.0
    v2 = (15.0 * ctp[4, :] - 327.0 * ctp[2, :] - 143.0) / 1152.0
    v3 = (-4042.0 * ctp[9, :] + 18189.0 * ctp[7, :] - 36387.0 * ctp[5, :] + 238425.0 * ctp[3, :] + 259290.0 * ct) / 414720.0
    v4 = (-121260.0 * ctp[10, :] + 551733.0 * ctp[8, :] - 151958.0 * ctp[6, :] - 57484425.0 * ctp[4, :] - 132752238.0 * ctp[2, :] - 12118727) / 39813120.0
    v5 = (82393456.0 * ctp[15, :] - 617950920.0 * ctp[13, :] + 2025529095.0 * ctp[11, :] - 3750839308.0 * ctp[9, :] + 3832454253.0 * ctp[7, :] + 35213253348.0 * ctp[5, :] + 130919230435.0 * ctp[3, :] + 34009066266 * ct) / 6688604160.0
    Ai, Aip, Bi, Bip = airy(mu ** (4.0 / 6.0) * zeta)
    P = 2.0 * sqrt(pi) * mu ** (1.0 / 6.0) * phi
    phip = phi ** arange(6, 31, 6).reshape((-1, 1))
    A0 = b0 * u0
    A1 = (b2 * u0 + phip[0, :] * b1 * u1 + phip[1, :] * b0 * u2) / zeta ** 3
    A2 = (b4 * u0 + phip[0, :] * b3 * u1 + phip[1, :] * b2 * u2 + phip[2, :] * b1 * u3 + phip[3, :] * b0 * u4) / zeta ** 6
    B0 = -(a1 * u0 + phip[0, :] * a0 * u1) / zeta ** 2
    B1 = -(a3 * u0 + phip[0, :] * a2 * u1 + phip[1, :] * a1 * u2 + phip[2, :] * a0 * u3) / zeta ** 5
    B2 = -(a5 * u0 + phip[0, :] * a4 * u1 + phip[1, :] * a3 * u2 + phip[2, :] * a2 * u3 + phip[3, :] * a1 * u4 + phip[4, :] * a0 * u5) / zeta ** 8
    U = P * (Ai * (A0 + A1 / mu ** 2.0 + A2 / mu ** 4.0) + Aip * (B0 + B1 / mu ** 2.0 + B2 / mu ** 4.0) / mu ** (8.0 / 6.0))
    Pd = sqrt(2.0 * pi) * mu ** (2.0 / 6.0) / phi
    C0 = -(b1 * v0 + phip[0, :] * b0 * v1) / zeta
    C1 = -(b3 * v0 + phip[0, :] * b2 * v1 + phip[1, :] * b1 * v2 + phip[2, :] * b0 * v3) / zeta ** 4
    C2 = -(b5 * v0 + phip[0, :] * b4 * v1 + phip[1, :] * b3 * v2 + phip[2, :] * b2 * v3 + phip[3, :] * b1 * v4 + phip[4, :] * b0 * v5) / zeta ** 7
    D0 = a0 * v0
    D1 = (a2 * v0 + phip[0, :] * a1 * v1 + phip[1, :] * a0 * v2) / zeta ** 3
    D2 = (a4 * v0 + phip[0, :] * a3 * v1 + phip[1, :] * a2 * v2 + phip[2, :] * a1 * v3 + phip[3, :] * a0 * v4) / zeta ** 6
    Ud = Pd * (Ai * (C0 + C1 / mu ** 2.0 + C2 / mu ** 4.0) / mu ** (4.0 / 6.0) + Aip * (D0 + D1 / mu ** 2.0 + D2 / mu ** 4.0))
    return (U, Ud)