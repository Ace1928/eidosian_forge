import copy
import datetime
from functools import partial
import logging
from pathlib import Path
from pickle import PicklingError
import pprint as pp
import traceback
from typing import (
import ray
from ray.exceptions import RpcError
from ray.train import CheckpointConfig, SyncConfig
from ray.train._internal.storage import StorageContext
from ray.tune.error import TuneError
from ray.tune.registry import register_trainable, is_function_trainable
from ray.tune.stopper import CombinedStopper, FunctionStopper, Stopper, TimeoutStopper
from ray.util.annotations import DeveloperAPI, Deprecated
def _convert_to_experiment_list(experiments: Union[Experiment, List[Experiment], Dict]):
    """Produces a list of Experiment objects.

    Converts input from dict, single experiment, or list of
    experiments to list of experiments. If input is None,
    will return an empty list.

    Arguments:
        experiments: Experiments to run.

    Returns:
        List of experiments.
    """
    exp_list = experiments
    if experiments is None:
        exp_list = []
    elif isinstance(experiments, Experiment):
        exp_list = [experiments]
    elif type(experiments) is dict:
        exp_list = [Experiment.from_json(name, spec) for name, spec in experiments.items()]
    if type(exp_list) is list and all((isinstance(exp, Experiment) for exp in exp_list)):
        if len(exp_list) > 1:
            logger.info('Running with multiple concurrent experiments. All experiments will be using the same SearchAlgorithm.')
    else:
        raise TuneError('Invalid argument: {}'.format(experiments))
    return exp_list