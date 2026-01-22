from fractions import Fraction
from typing import Dict, Any, List, Tuple
import numpy as np
from qiskit.pulse import instructions
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def _phase_to_text(formatter: Dict[str, Any], phase: float, max_denom: int=10, flip: bool=True) -> Tuple[str, str]:
    """A helper function to convert a float value to text with pi.

    Args:
        formatter: Dictionary of stylesheet settings.
        phase: A phase value in units of rad.
        max_denom: Maximum denominator. Return raw value if exceed.
        flip: Set `True` to flip the sign.

    Returns:
        Standard text and latex text of phase value.
    """
    try:
        phase = float(phase)
    except TypeError:
        return (formatter['unicode_symbol.phase_parameter'], formatter['latex_symbol.phase_parameter'])
    frac = Fraction(np.abs(phase) / np.pi)
    if phase == 0:
        return ('0', '0')
    num = frac.numerator
    denom = frac.denominator
    if denom > max_denom:
        latex = f'{np.abs(phase):.2f}'
        plain = f'{np.abs(phase):.2f}'
    elif num == 1:
        if denom == 1:
            latex = '\\pi'
            plain = 'pi'
        else:
            latex = f'\\pi/{denom:d}'
            plain = f'pi/{denom:d}'
    else:
        latex = f'{num:d}/{denom:d} \\pi'
        plain = f'{num:d}/{denom:d} pi'
    if flip:
        sign = '-' if phase > 0 else ''
    else:
        sign = '-' if phase < 0 else ''
    return (sign + plain, sign + latex)