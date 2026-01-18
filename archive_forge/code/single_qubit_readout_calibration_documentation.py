import dataclasses
import time
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING
import sympy
import numpy as np
from cirq import circuits, ops, study
Estimate single qubit readout error using parallel operations.

    For each trial, prepare and then measure a random computational basis
    bitstring on qubits using gates in parallel.
    Returns a SingleQubitReadoutCalibrationResult which can be used to
    compute readout errors for each qubit.

    Args:
        sampler: The `cirq.Sampler` used to run the circuits.
        qubits: The qubits being tested.
        repetitions: The number of measurement repetitions to perform for
            each trial.
        trials: The number of bitstrings to prepare.
        trials_per_batch:  If provided, split the experiment into batches
            with this number of trials in each batch.
        bit_strings: Optional numpy array of shape (trials, qubits) where the
            first dimension is the number of the trial and the second
            dimension is the qubit (ordered by the qubit order from
            the qubits parameter).  Each value should be a 0 or 1 which
            specifies which state the qubit should be prepared into during
            that trial.  If not provided, the function will generate random
            bit strings for you.

    Returns:
        A SingleQubitReadoutCalibrationResult storing the readout error
        probabilities as well as the number of repetitions used to estimate
        the probabilities. Also stores a timestamp indicating the time when
        data was finished being collected from the sampler.  Note that,
        if there did not exist a trial where a given qubit was set to |0〉,
        the zero-state error will be set to `nan` (not a number).  Likewise
        for qubits with no |1〉trial and one-state error.

    Raises:
        ValueError: If the number of trials, repetitions, or trials_per batch is
            negative, or if bit_strings is not a numpy array or of the wrong
            shape.
    