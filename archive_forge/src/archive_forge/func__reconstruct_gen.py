from functools import wraps
from inspect import signature
import warnings
import numpy as np
from autoray import numpy as anp
import pennylane as qml
def _reconstruct_gen(fun, spectrum, shifts=None, x0=None, f0=None, interface=None):
    """Reconstruct a univariate (real-valued) Fourier series with given spectrum.

    Args:
        fun (callable): Univariate finite Fourier series to reconstruct.
            It must have signature ``float -> float`` .
        spectrum (Collection): Frequency spectrum of the Fourier series;
            non-positive frequencies are ignored.
        shifts (Sequence): Shift angles at which to evaluate ``fun`` for the reconstruction.
            Chosen equidistantly within the interval :math:`[0, 2\\pi/f_\\text{max}]`
            if ``shifts=None`` , where :math:`f_\\text{max}` is the biggest
            frequency in ``spectrum``.
        x0 (float): Center to which to shift the reconstruction.
            The points at which ``fun`` is evaluated are *not* affected
            by ``x0`` .
        f0 (float): Value of ``fun`` at zero; If :math:`0` is among the ``shifts``
            and ``f0`` is provided, one evaluation of ``fun`` is saved.
        interface (str): Which auto-differentiation framework to use as
            interface. This determines in which interface the output
            reconstructed function is intended to be used.

    Returns:
        callable: Reconstructed Fourier series with :math:`R` frequencies in ``spectrum`` .
        This function is a purely classical function. Furthermore, it is fully differentiable.
    """
    have_f0 = f0 is not None
    have_shifts = shifts is not None
    spectrum = anp.asarray(spectrum, like=interface)
    spectrum = spectrum[spectrum > 0]
    f_max = qml.math.max(spectrum)
    if not have_shifts:
        R = qml.math.shape(spectrum)[0]
        shifts = qml.math.arange(-R, R + 1) * 2 * np.pi / (f_max * (2 * R + 1)) * R
        zero_idx = R
        need_f0 = True
    elif have_f0:
        zero_idx = qml.math.where(qml.math.isclose(shifts, qml.math.zeros_like(shifts[0])))
        zero_idx = zero_idx[0][0] if len(zero_idx) > 0 and len(zero_idx[0]) > 0 else None
        need_f0 = zero_idx is not None
    if have_f0 and need_f0:
        shifts = qml.math.concatenate([shifts[zero_idx:zero_idx + 1], shifts[:zero_idx], shifts[zero_idx + 1:]])
        shifts = anp.asarray(shifts, like=interface)
        evals = anp.asarray([f0] + list(map(fun, shifts[1:])), like=interface)
    else:
        shifts = anp.asarray(shifts, like=interface)
        if have_f0 and (not need_f0):
            warnings.warn(_warn_text_f0_ignored)
        evals = anp.asarray(list(map(fun, shifts)), like=interface)
    L = len(shifts)
    C1 = qml.math.ones((L, 1))
    C2 = qml.math.cos(qml.math.tensordot(shifts, spectrum, axes=0))
    C3 = qml.math.sin(qml.math.tensordot(shifts, spectrum, axes=0))
    C = qml.math.hstack([C1, C2, C3])
    cond = qml.math.linalg.cond(C)
    if cond > 100000000.0:
        warnings.warn(f'The condition number of the Fourier transform matrix is very large: {cond}.', UserWarning)
    W = qml.math.linalg.solve(C, evals)
    R = (L - 1) // 2
    a0 = W[0]
    a = anp.asarray(W[1:R + 1], like=interface)
    b = anp.asarray(W[R + 1:], like=interface)
    x0 = anp.asarray(np.float64(0.0), like=interface) if x0 is None else x0

    def _reconstruction(x):
        """Univariate reconstruction based on arbitrary shifts."""
        x = x - x0
        return a0 + qml.math.tensordot(qml.math.cos(spectrum * x), a, axes=[[0], [0]]) + qml.math.tensordot(qml.math.sin(spectrum * x), b, axes=[[0], [0]])
    return _reconstruction