import collections
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import duet
import pandas as pd
from cirq import ops, protocols, study, value
from cirq.work.observable_measurement import (
from cirq.work.observable_settings import _hashable_param
def _run_sweep_impl(self, program: 'cirq.AbstractCircuit', params: 'cirq.Sweepable', repetitions: int=1) -> Sequence['cirq.Result']:
    """Implements run_sweep using run_sweep_async"""
    return duet.run(self.run_sweep_async, program, params, repetitions)