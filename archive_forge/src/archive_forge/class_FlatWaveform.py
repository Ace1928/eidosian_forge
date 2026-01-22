from copy import copy
from dataclasses import dataclass
from numbers import Complex, Real
from typing import Callable, Dict, Union, List, Optional, no_type_check
import numpy as np
from scipy.special import erf
from pyquil.quilatom import TemplateWaveform, _update_envelope, _complex_str, Expression, substitute
@waveform('flat')
class FlatWaveform(TemplateWaveform):
    """
    A flat (constant) waveform.
    """
    iq: Complex
    ' A raw IQ value. '
    scale: Optional[float] = None
    ' An optional global scaling factor. '
    phase: Optional[float] = None
    ' An optional phase shift factor. '
    detuning: Optional[float] = None
    ' An optional frequency detuning factor. '

    def out(self) -> str:
        output = 'flat('
        output += ', '.join([f'duration: {self.duration}', f'iq: {_complex_str(self.iq)}'] + _optional_field_strs(self))
        output += ')'
        return output

    def __str__(self) -> str:
        return self.out()

    def samples(self, rate: float) -> np.ndarray:
        iqs = np.full(self.num_samples(rate), self.iq, dtype=np.complex128)
        return _update_envelope(iqs, rate, scale=self.scale, phase=self.phase, detuning=self.detuning)