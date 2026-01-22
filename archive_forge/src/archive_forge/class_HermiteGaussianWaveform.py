import sys
from warnings import warn
from rpcq._base import Message
from typing import Any, List, Dict, Optional
from rpcq.messages import *
@dataclass(eq=False, repr=False)
class HermiteGaussianWaveform(TemplateWaveform):
    """
    Hermite-Gaussian shaped pulse. Reference: Effects of arbitrary laser
      or NMR pulse shapes on population inversion and coherence Warren S. Warren.
      81, (1984); doi: 10.1063/1.447644
    """
    fwhm: Optional[float] = None
    'Full Width Half Max shape paramter in seconds'
    t0: float = 0.0
    'Center time coordinate of the shape in seconds. Defaults to mid-point of pulse.'
    anh: float = -210000000.0
    'Anharmonicity of the qubit, f01-f12 in Hz'
    alpha: float = 0.0
    'Dimensionless DRAG parameter'
    second_order_hrm_coeff: float = 0.956
    'Second order coefficient (see paper)'