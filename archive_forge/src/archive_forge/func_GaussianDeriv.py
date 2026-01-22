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
def GaussianDeriv(duration: int | ParameterValueType, amp: float | ParameterExpression, sigma: float | ParameterExpression, angle: float | ParameterExpression | None=0.0, name: str | None=None, limit_amplitude: bool | None=None) -> ScalableSymbolicPulse:
    """An unnormalized Gaussian derivative pulse.

    The Gaussian function is centered around the halfway point of the pulse,
    and the envelope of the pulse is given by:

    .. math::

        f(x) = -\\text{A}\\frac{x-\\mu}{\\text{sigma}^{2}}\\exp
            \\left[-\\left(\\frac{x-\\mu}{2\\text{sigma}}\\right)^{2}\\right]  ,  0 <= x < duration

    where :math:`\\text{A} = \\text{amp} \\times\\exp\\left(i\\times\\text{angle}\\right)`,
    and :math:`\\mu=\\text{duration}/2`.

    Args:
        duration: Pulse length in terms of the sampling period `dt`.
        amp: The magnitude of the amplitude of the pulse
            (the value of the corresponding Gaussian at the midpoint `duration`/2).
        sigma: A measure of how wide or narrow the corresponding Gaussian peak is in terms of `dt`;
            described mathematically in the class docstring.
        angle: The angle in radians of the complex phase factor uniformly
            scaling the pulse. Default value 0.
        name: Display name for this pulse envelope.
        limit_amplitude: If ``True``, then limit the amplitude of the
            waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

    Returns:
        ScalableSymbolicPulse instance.
    """
    parameters = {'sigma': sigma}
    _t, _duration, _amp, _angle, _sigma = sym.symbols('t, duration, amp, angle, sigma')
    envelope_expr = -_amp * sym.exp(sym.I * _angle) * ((_t - _duration / 2) / _sigma ** 2) * sym.exp(-(1 / 2) * ((_t - _duration / 2) / _sigma) ** 2)
    consts_expr = _sigma > 0
    valid_amp_conditions_expr = sym.Abs(_amp / _sigma) <= sym.exp(1 / 2)
    return ScalableSymbolicPulse(pulse_type='GaussianDeriv', duration=duration, amp=amp, angle=angle, parameters=parameters, name=name, limit_amplitude=limit_amplitude, envelope=envelope_expr, constraints=consts_expr, valid_amp_conditions=valid_amp_conditions_expr)