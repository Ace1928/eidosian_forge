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
def _parse_checkpoint_options(checkpoint: bool, checkpoint_fn: Optional[str], checkpoint_other_fn: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Parse the checkpoint-oriented options in `measure_grouped_settings`.

    This function contains the validation and defaults logic. Please see
    `measure_grouped_settings` for documentation on these args.

    Returns:
        checkpoint_fn, checkpoint_other_fn: Parsed or default filenames for primary and previous
            checkpoint files.

    Raises:
        ValueError: If a `checkpoint_fn` was specified, but `checkpoint` was False, if the
            `checkpoint_fn` is not of the form filename.json, or if `checkout_fn` and
            `checkpoint_other_fn` are the same filename.
    """
    if not checkpoint:
        if checkpoint_fn is not None or checkpoint_other_fn is not None:
            raise ValueError('Checkpoint filenames were provided but `checkpoint` was set to False.')
        return (None, None)
    if checkpoint_fn is None:
        checkpoint_dir = tempfile.mkdtemp()
        chk_basename = 'observables'
        checkpoint_fn = f'{checkpoint_dir}/{chk_basename}.json'
    if checkpoint_other_fn is None:
        checkpoint_dir = os.path.dirname(checkpoint_fn)
        chk_basename = os.path.basename(checkpoint_fn)
        chk_basename, dot, ext = chk_basename.rpartition('.')
        if chk_basename == '' or dot != '.' or ext == '':
            raise ValueError(f"You specified `checkpoint_fn={checkpoint_fn!r}` which does not follow the pattern of 'filename.extension'. Please follow this pattern or fully specify `checkpoint_other_fn`.")
        if ext != 'json':
            raise ValueError('Please use a `.json` filename or fully specify checkpoint_fn and checkpoint_other_fn')
        if checkpoint_dir == '':
            checkpoint_other_fn = f'{chk_basename}.prev.json'
        else:
            checkpoint_other_fn = f'{checkpoint_dir}/{chk_basename}.prev.json'
    if checkpoint_fn == checkpoint_other_fn:
        raise ValueError(f'`checkpoint_fn` and `checkpoint_other_fn` were set to the same filename: {checkpoint_fn}. Please use two different filenames.')
    return (checkpoint_fn, checkpoint_other_fn)