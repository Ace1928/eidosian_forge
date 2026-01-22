from typing import List
import pytest
import sympy
import numpy as np
import cirq
import cirq_google as cg
def _batch_size_less_than_two(circuits: List[cirq.Circuit], sweeps: List[cirq.Sweepable], repetitions: int):
    if len(circuits) > 2:
        raise ValueError('Too many batches')