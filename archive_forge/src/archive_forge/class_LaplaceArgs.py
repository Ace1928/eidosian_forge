import os
from enum import Enum, auto
from time import time
from typing import Any, Dict, List, Mapping, Optional, Union
import numpy as np
from numpy.random import default_rng
from cmdstanpy import _TMPDIR
from cmdstanpy.utils import (
class LaplaceArgs:
    """Arguments needed for laplace method."""

    def __init__(self, mode: str, draws: Optional[int]=None, jacobian: bool=True) -> None:
        self.mode = mode
        self.jacobian = jacobian
        self.draws = draws

    def validate(self, _chains: Optional[int]=None) -> None:
        """Check arguments correctness and consistency."""
        if not os.path.exists(self.mode):
            raise ValueError(f'Invalid path for mode file: {self.mode}')
        positive_int(self.draws, 'draws')

    def compose(self, _idx: int, cmd: List[str]) -> List[str]:
        """compose command string for CmdStan for non-default arg values."""
        cmd.append('method=laplace')
        cmd.append(f'mode={self.mode}')
        if self.draws:
            cmd.append(f'draws={self.draws}')
        if not self.jacobian:
            cmd.append('jacobian=0')
        return cmd