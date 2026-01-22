import abc
import dataclasses
import itertools
import os
import tempfile
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, study, ops, value, protocols
from cirq._doc import document
from cirq.work.observable_grouping import group_settings_greedy, GROUPER_T
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import InitObsSetting, observables_to_settings, _MeasurementSpec
def _check_meas_specs_still_todo(meas_specs: List[_MeasurementSpec], accumulators: Dict[_MeasurementSpec, BitstringAccumulator], stopping_criteria: StoppingCriteria) -> Tuple[List[_MeasurementSpec], int]:
    """Filter `meas_specs` in case some are done.

    In the sampling loop in `measure_grouped_settings`, we submit
    each `meas_spec` in chunks. This function contains the logic for
    removing `meas_spec`s from the loop if they are done.
    """
    still_todo = []
    repetitions_set: Set[int] = set()
    for meas_spec in meas_specs:
        accumulator = accumulators[meas_spec]
        more_repetitions = stopping_criteria.more_repetitions(accumulator)
        if more_repetitions < 0:
            raise ValueError("Stopping criteria's `more_repetitions` should return 0 or a positive number.")
        if more_repetitions == 0:
            continue
        repetitions_set.add(more_repetitions)
        still_todo.append(meas_spec)
    if len(still_todo) == 0:
        return (still_todo, 0)
    repetitions = _aggregate_n_repetitions(repetitions_set)
    total_repetitions = len(still_todo) * repetitions
    if total_repetitions > MAX_REPETITIONS_PER_JOB:
        old_repetitions = repetitions
        repetitions = MAX_REPETITIONS_PER_JOB // len(still_todo)
        if repetitions < 10:
            raise ValueError('You have requested too many parameter settings to batch your job effectively. Consider fewer sweeps or manually splitting sweeps into multiple jobs.')
        warnings.warn(f'The number of requested sweep parameters is high. To avoid a batched job with more than {MAX_REPETITIONS_PER_JOB} shots, the number of shots per call to run_sweep (per parameter value) will be throttled from {old_repetitions} to {repetitions}.')
    return (still_todo, repetitions)