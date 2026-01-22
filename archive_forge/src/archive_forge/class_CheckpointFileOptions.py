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
@dataclasses.dataclass(frozen=True)
class CheckpointFileOptions:
    """Options to configure "checkpointing" to save intermediate results.

    Args:
        checkpoint: If set to True, save cumulative raw results at the end
            of each iteration of the sampling loop. Load in these results
            with `cirq.read_json`.
        checkpoint_fn: The filename for the checkpoint file. If `checkpoint`
            is set to True and this is not specified, a file in a temporary
            directory will be used.
        checkpoint_other_fn: The filename for another checkpoint file, which
            contains the previous checkpoint. This lets us avoid losing data if
            a failure occurs during checkpoint writing. If `checkpoint`
            is set to True and this is not specified, a file in a temporary
            directory will be used. If `checkpoint` is set to True and
            `checkpoint_fn` is specified but this argument is *not* specified,
            "{checkpoint_fn}.prev.json" will be used.
    """
    checkpoint: bool = False
    checkpoint_fn: Optional[str] = None
    checkpoint_other_fn: Optional[str] = None

    def __post_init__(self):
        fn, other_fn = _parse_checkpoint_options(self.checkpoint, self.checkpoint_fn, self.checkpoint_other_fn)
        object.__setattr__(self, 'checkpoint_fn', fn)
        object.__setattr__(self, 'checkpoint_other_fn', other_fn)

    def maybe_to_json(self, obj: Any):
        """Call `cirq.to_json with `value` according to the configuration options in this class.

        If `checkpoint=False`, nothing will happen. Otherwise, we will use `checkpoint_fn` and
        `checkpoint_other_fn` as the destination JSON file as described in the class docstring.
        """
        if not self.checkpoint:
            return
        assert self.checkpoint_fn is not None, 'mypy'
        assert self.checkpoint_other_fn is not None, 'mypy'
        if os.path.exists(self.checkpoint_fn):
            os.replace(self.checkpoint_fn, self.checkpoint_other_fn)
        protocols.to_json(obj, self.checkpoint_fn)