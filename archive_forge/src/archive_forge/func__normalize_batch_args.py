import collections
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import duet
import pandas as pd
from cirq import ops, protocols, study, value
from cirq.work.observable_measurement import (
from cirq.work.observable_settings import _hashable_param
def _normalize_batch_args(self, programs: Sequence['cirq.AbstractCircuit'], params_list: Optional[Sequence['cirq.Sweepable']]=None, repetitions: Union[int, Sequence[int]]=1) -> Tuple[Sequence['cirq.Sweepable'], Sequence[int]]:
    if params_list is None:
        params_list = [None] * len(programs)
    if len(programs) != len(params_list):
        raise ValueError(f'len(programs) and len(params_list) must match. Got {len(programs)} and {len(params_list)}.')
    if isinstance(repetitions, int):
        repetitions = [repetitions] * len(programs)
    if len(programs) != len(repetitions):
        raise ValueError(f'len(programs) and len(repetitions) must match. Got {len(programs)} and {len(repetitions)}.')
    return (params_list, repetitions)