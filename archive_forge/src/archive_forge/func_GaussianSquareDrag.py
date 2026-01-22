from __future__ import annotations
import functools
import warnings
from collections.abc import Mapping, Callable
from copy import deepcopy
from typing import Any
import numpy as np
import symengine as sym
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library.pulse import Pulse
from qiskit.pulse.library.waveform import Waveform
def GaussianSquareDrag(duration: int | ParameterExpression, amp: float | ParameterExpression, sigma: float | ParameterExpression, beta: float | ParameterExpression, width: float | ParameterExpression | None=None, angle: float | ParameterExpression | None=0.0, risefall_sigma_ratio: float | ParameterExpression | None=None, name: str | None=None, limit_amplitude: bool | None=None) -> ScalableSymbolicPulse:
    """A square pulse with a Drag shaped rise and fall

    This pulse shape is similar to :class:`~.GaussianSquare` but uses
    :class:`~.Drag` for its rise and fall instead of :class:`~.Gaussian`. The
    addition of the DRAG component of the rise and fall is sometimes helpful in
    suppressing the spectral content of the pulse at frequencies near to, but
    slightly offset from, the fundamental frequency of the drive. When there is
    a spectator qubit close in frequency to the fundamental frequency,
    suppressing the drive at the spectator's frequency can help avoid unwanted
    excitation of the spectator.

    Exactly one of the ``risefall_sigma_ratio`` and ``width`` parameters has to be specified.

    If ``risefall_sigma_ratio`` is not ``None`` and ``width`` is ``None``:

    .. math::

        \\begin{aligned}
        \\text{risefall} &= \\text{risefall\\_sigma\\_ratio} \\times \\text{sigma}\\\\
        \\text{width} &= \\text{duration} - 2 \\times \\text{risefall}
        \\end{aligned}

    If ``width`` is not None and ``risefall_sigma_ratio`` is None:

    .. math:: \\text{risefall} = \\frac{\\text{duration} - \\text{width}}{2}

    Gaussian :math:`g(x, c, σ)` and lifted gaussian :math:`g'(x, c, σ)` curves
    can be written as:

    .. math::

        \\begin{aligned}
        g(x, c, σ) &= \\exp\\Bigl(-\\frac12 \\frac{(x - c)^2}{σ^2}\\Bigr)\\\\
        g'(x, c, σ) &= \\frac{g(x, c, σ)-g(-1, c, σ)}{1-g(-1, c, σ)}
        \\end{aligned}

    From these, the lifted DRAG curve :math:`d'(x, c, σ, β)` can be written as

    .. math::

        d'(x, c, σ, β) = g'(x, c, σ) \\times \\Bigl(1 + 1j \\times β \\times            \\Bigl(-\\frac{x - c}{σ^2}\\Bigr)\\Bigr)

    The lifted gaussian square drag pulse :math:`f'(x)` is defined as:

    .. math::

        \\begin{aligned}
        f'(x) &= \\begin{cases}            \\text{A} \\times d'(x, \\text{risefall}, \\text{sigma}, \\text{beta})                & x < \\text{risefall}\\\\
            \\text{A}                & \\text{risefall} \\le x < \\text{risefall} + \\text{width}\\\\
            \\text{A} \\times \\times d'(                    x - (\\text{risefall} + \\text{width}),                    \\text{risefall},                    \\text{sigma},                    \\text{beta}                )                & \\text{risefall} + \\text{width} \\le x        \\end{cases}\\\\
        \\end{aligned}

    where :math:`\\text{A} = \\text{amp} \\times
    \\exp\\left(i\\times\\text{angle}\\right)`.

    Args:
        duration: Pulse length in terms of the sampling period `dt`.
        amp: The amplitude of the DRAG rise and fall and of the square pulse.
        sigma: A measure of how wide or narrow the DRAG risefall is; see the class
               docstring for more details.
        beta: The DRAG correction amplitude.
        width: The duration of the embedded square pulse.
        angle: The angle in radians of the complex phase factor uniformly
            scaling the pulse. Default value 0.
        risefall_sigma_ratio: The ratio of each risefall duration to sigma.
        name: Display name for this pulse envelope.
        limit_amplitude: If ``True``, then limit the amplitude of the
            waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

    Returns:
        ScalableSymbolicPulse instance.

    Raises:
        PulseError: When width and risefall_sigma_ratio are both empty or both non-empty.
    """
    if width is None and risefall_sigma_ratio is None:
        raise PulseError('Either the pulse width or the risefall_sigma_ratio parameter must be specified.')
    if width is not None and risefall_sigma_ratio is not None:
        raise PulseError('Either the pulse width or the risefall_sigma_ratio parameter can be specified but not both.')
    if width is None and risefall_sigma_ratio is not None:
        width = duration - 2.0 * risefall_sigma_ratio * sigma
    parameters = {'sigma': sigma, 'width': width, 'beta': beta}
    _t, _duration, _amp, _sigma, _beta, _width, _angle = sym.symbols('t, duration, amp, sigma, beta, width, angle')
    _center = _duration / 2
    _sq_t0 = _center - _width / 2
    _sq_t1 = _center + _width / 2
    _gaussian_ledge = _lifted_gaussian(_t, _sq_t0, -1, _sigma)
    _gaussian_redge = _lifted_gaussian(_t, _sq_t1, _duration + 1, _sigma)
    _deriv_ledge = -(_t - _sq_t0) / _sigma ** 2 * _gaussian_ledge
    _deriv_redge = -(_t - _sq_t1) / _sigma ** 2 * _gaussian_redge
    envelope_expr = _amp * sym.exp(sym.I * _angle) * sym.Piecewise((_gaussian_ledge + sym.I * _beta * _deriv_ledge, _t <= _sq_t0), (_gaussian_redge + sym.I * _beta * _deriv_redge, _t >= _sq_t1), (1, True))
    consts_expr = sym.And(_sigma > 0, _width >= 0, _duration >= _width)
    valid_amp_conditions_expr = sym.And(sym.Abs(_amp) <= 1.0, sym.Abs(_beta) < _sigma)
    return ScalableSymbolicPulse(pulse_type='GaussianSquareDrag', duration=duration, amp=amp, angle=angle, parameters=parameters, name=name, limit_amplitude=limit_amplitude, envelope=envelope_expr, constraints=consts_expr, valid_amp_conditions=valid_amp_conditions_expr)