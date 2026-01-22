from copy import copy
from dataclasses import dataclass
from numbers import Complex, Real
from typing import Callable, Dict, Union, List, Optional, no_type_check
import numpy as np
from scipy.special import erf
from pyquil.quilatom import TemplateWaveform, _update_envelope, _complex_str, Expression, substitute
@waveform('drag_gaussian')
class DragGaussianWaveform(TemplateWaveform):
    """A DRAG Gaussian pulse."""
    fwhm: float
    ' The Full-Width-Half-Max of the gaussian (seconds). '
    t0: float
    ' The center time coordinate of the Gaussian (seconds). '
    anh: float
    ' The anharmonicity of the qubit, f01-f12 (Hertz). '
    alpha: float
    ' Dimensionles DRAG parameter. '
    scale: Optional[float] = None
    ' An optional global scaling factor. '
    phase: Optional[float] = None
    ' An optional phase shift factor. '
    detuning: Optional[float] = None
    ' An optional frequency detuning factor. '

    def out(self) -> str:
        output = 'drag_gaussian('
        output += ', '.join([f'duration: {self.duration}', f'fwhm: {self.fwhm}', f't0: {self.t0}', f'anh: {self.anh}', f'alpha: {self.alpha}'] + _optional_field_strs(self))
        output += ')'
        return output

    def __str__(self) -> str:
        return self.out()

    def samples(self, rate: float) -> np.ndarray:
        ts = np.arange(self.num_samples(rate), dtype=np.complex128) / rate
        sigma = 0.5 * self.fwhm / np.sqrt(2.0 * np.log(2.0))
        env = np.exp(-0.5 * (ts - self.t0) ** 2 / sigma ** 2)
        env_der = self.alpha * (1.0 / (2 * np.pi * self.anh * sigma ** 2)) * (ts - self.t0) * env
        iqs = env + 1j * env_der
        return _update_envelope(iqs, rate, scale=self.scale, phase=self.phase, detuning=self.detuning)