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
def Cos(duration: int | ParameterExpression, amp: float | ParameterExpression, phase: float | ParameterExpression, freq: float | ParameterExpression | None=None, angle: float | ParameterExpression | None=0.0, name: str | None=None, limit_amplitude: bool | None=None) -> ScalableSymbolicPulse:
    """A cosine pulse.

    The envelope of the pulse is given by:

    .. math::

        f(x) = \\text{A}\\cos\\left(2\\pi\\text{freq}x+\\text{phase}\\right)  ,  0 <= x < duration

    where :math:`\\text{A} = \\text{amp} \\times\\exp\\left(i\\times\\text{angle}\\right)`.

    Args:
        duration: Pulse length in terms of the sampling period `dt`.
        amp: The magnitude of the amplitude of the cosine wave. Wave range is [-`amp`,`amp`].
        phase: The phase of the cosine wave (note that this is not equivalent to the angle
            of the complex amplitude).
        freq: The frequency of the cosine wave, in terms of 1 over sampling period.
            If not provided defaults to a single cycle (i.e :math:'\\frac{1}{\\text{duration}}').
            The frequency is limited to the range :math:`\\left(0,0.5\\right]` (the Nyquist frequency).
        angle: The angle in radians of the complex phase factor uniformly
            scaling the pulse. Default value 0.
        name: Display name for this pulse envelope.
        limit_amplitude: If ``True``, then limit the amplitude of the
            waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

    Returns:
        ScalableSymbolicPulse instance.
    """
    if freq is None:
        freq = 1 / duration
    parameters = {'freq': freq, 'phase': phase}
    _t, _duration, _amp, _angle, _freq, _phase = sym.symbols('t, duration, amp, angle, freq, phase')
    envelope_expr = _amp * sym.exp(sym.I * _angle) * sym.cos(2 * sym.pi * _freq * _t + _phase)
    consts_expr = sym.And(_freq > 0, _freq < 0.5)
    valid_amp_conditions_expr = sym.Abs(_amp) <= 1.0
    return ScalableSymbolicPulse(pulse_type='Cos', duration=duration, amp=amp, angle=angle, parameters=parameters, name=name, limit_amplitude=limit_amplitude, envelope=envelope_expr, constraints=consts_expr, valid_amp_conditions=valid_amp_conditions_expr)