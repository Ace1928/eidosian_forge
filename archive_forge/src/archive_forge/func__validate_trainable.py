import copy
import io
import os
import math
import logging
from pathlib import Path
from typing import (
import pyarrow.fs
import ray.cloudpickle as pickle
from ray.util import inspect_serializability
from ray.air._internal.uri_utils import URI
from ray.air._internal.usage import AirEntrypoint
from ray.air.config import RunConfig, ScalingConfig
from ray.train._internal.storage import StorageContext, get_fs_and_path
from ray.tune import Experiment, TuneError, ExperimentAnalysis
from ray.tune.execution.experiment_state import _ResumeConfig
from ray.tune.tune import _Config
from ray.tune.registry import is_function_trainable
from ray.tune.result import _get_defaults_results_dir
from ray.tune.result_grid import ResultGrid
from ray.tune.trainable import Trainable
from ray.tune.tune import run
from ray.tune.tune_config import TuneConfig
from ray.tune.utils import flatten_dict
def _validate_trainable(self, trainable: TrainableType, required_trainable_name: Optional[str]=None):
    """Determines whether or not the trainable is valid.

        This includes checks on the serializability of the trainable, as well
        asserting that the trainable name is as expected on restoration.

        This trainable name validation is needed due to an implementation detail
        where the trainable name (which is differently generated depending on
        the trainable type) is saved in the Trial metadata and needs to match
        upon restoration. This does not affect the typical path, since `Tuner.restore`
        expects the exact same trainable (which will have the same name).

        Raises:
            ValueError: if the trainable name does not match or if the trainable
                is not serializable.
        """
    try:
        pickle.dumps(trainable)
    except TypeError as e:
        sio = io.StringIO()
        inspect_serializability(trainable, print_file=sio)
        msg = f'The provided trainable is not serializable, which is a requirement since the trainable is serialized and deserialized when transferred to remote workers. See below for a trace of the non-serializable objects that were found in your trainable:\n{sio.getvalue()}'
        raise TypeError(msg) from e
    if not required_trainable_name:
        return
    trainable_name = Experiment.get_trainable_name(trainable)
    if trainable_name != required_trainable_name:
        raise ValueError(f"Invalid `trainable` input to `Tuner.restore()`. To fix this error, pass in the same trainable that was used to initialize the Tuner. Got a trainable with identifier '{trainable_name}' but expected '{required_trainable_name}'.")