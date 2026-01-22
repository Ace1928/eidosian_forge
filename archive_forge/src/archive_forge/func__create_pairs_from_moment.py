import abc
import collections
import dataclasses
import functools
import math
import re
from typing import (
import numpy as np
import pandas as pd
import cirq
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.ops import FSimGateFamily, SycamoreGate
def _create_pairs_from_moment(moment: cirq.Moment) -> Tuple[Tuple[Tuple[cirq.Qid, cirq.Qid], ...], cirq.Gate]:
    """Creates instantiation parameters from a Moment.

    Given a moment, creates a tuple of pairs of qubits and the
    gate for instantiation of a sub-class of PhasedFSimCalibrationRequest,
    Sub-classes of PhasedFSimCalibrationRequest can call this function
    to implement a from_moment function.
    """
    gate = None
    pairs: List[Tuple[cirq.Qid, cirq.Qid]] = []
    for op in moment:
        if op.gate is None:
            raise ValueError('All gates in request object must be two qubit gates: {op}')
        if gate is None:
            gate = op.gate
        elif gate != op.gate:
            raise ValueError('All gates in request object must be identical {gate}!={op.gate}')
        if len(op.qubits) != 2:
            raise ValueError('All gates in request object must be two qubit gates: {op}')
        pairs.append((op.qubits[0], op.qubits[1]))
    if gate is None:
        raise ValueError('No gates found to create request {moment}')
    return (tuple(pairs), gate)