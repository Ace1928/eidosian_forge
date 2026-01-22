import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def _generate_wave_table(wave_type: str, data_type: str, table_size: int, min: float, max: float, phase: float, device: torch.device) -> Tensor:
    """A helper function for phaser. Generates a table with given parameters.

    Args:
        wave_type (str): SINE or TRIANGULAR
        data_type (str): desired data_type ( `INT` or `FLOAT` )
        table_size (int): desired table size
        min (float): desired min value
        max (float): desired max value
        phase (float): desired phase
        device (torch.device): Torch device on which table must be generated
    Returns:
        Tensor: A 1D tensor with wave table values
    """
    phase_offset = int(phase / math.pi / 2 * table_size + 0.5)
    t = torch.arange(table_size, device=device, dtype=torch.int32)
    point = (t + phase_offset) % table_size
    d = torch.zeros_like(point, device=device, dtype=torch.float64)
    if wave_type == 'SINE':
        d = (torch.sin(point.to(torch.float64) / table_size * 2 * math.pi) + 1) / 2
    elif wave_type == 'TRIANGLE':
        d = point.to(torch.float64) * 2 / table_size
        value = torch.div(4 * point, table_size, rounding_mode='floor')
        d[value == 0] = d[value == 0] + 0.5
        d[value == 1] = 1.5 - d[value == 1]
        d[value == 2] = 1.5 - d[value == 2]
        d[value == 3] = d[value == 3] - 1.5
    d = d * (max - min) + min
    if data_type == 'INT':
        mask = d < 0
        d[mask] = d[mask] - 0.5
        d[~mask] = d[~mask] + 0.5
        d = d.to(torch.int32)
    elif data_type == 'FLOAT':
        d = d.to(torch.float32)
    return d