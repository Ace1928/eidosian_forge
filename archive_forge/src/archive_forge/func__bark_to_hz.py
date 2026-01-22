import math
import warnings
from typing import Optional
import torch
from torchaudio.functional.functional import _create_triangular_filterbank
def _bark_to_hz(barks: torch.Tensor, bark_scale: str='traunmuller') -> torch.Tensor:
    """Convert bark bin numbers to frequencies.

    Args:
        barks (torch.Tensor): Bark frequencies
        bark_scale (str, optional): Scale to use: ``traunmuller``,``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        freqs (torch.Tensor): Barks converted in Hz
    """
    if bark_scale not in ['schroeder', 'traunmuller', 'wang']:
        raise ValueError('bark_scale should be one of "traunmuller", "schroeder" or "wang".')
    if bark_scale == 'wang':
        return 600.0 * torch.sinh(barks / 6.0)
    elif bark_scale == 'schroeder':
        return 650.0 * torch.sinh(barks / 7.0)
    if any(barks < 2):
        idx = barks < 2
        barks[idx] = (barks[idx] - 0.3) / 0.85
    elif any(barks > 20.1):
        idx = barks > 20.1
        barks[idx] = (barks[idx] + 4.422) / 1.22
    freqs = 1960 * ((barks + 0.53) / (26.28 - barks))
    return freqs