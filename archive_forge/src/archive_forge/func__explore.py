import copy
import json
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from ray.air.constants import TRAINING_ITERATION
from ray.train import Checkpoint
from ray.train._internal.session import _TrainingResult, _FutureTrainingResult
from ray.tune.error import TuneError
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search import SearchGenerator
from ray.tune.utils.util import SafeFallbackEncoder
from ray.tune.search.sample import Domain, Function
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
from ray.tune.search.variant_generator import format_vars
from ray.tune.experiment import Trial
from ray.util import PublicAPI
from ray.util.debug import log_once
def _explore(config: Dict, mutations: Dict, resample_probability: float, perturbation_factors: Tuple[float], custom_explore_fn: Optional[Callable]) -> Tuple[Dict, Dict]:
    """Return a perturbed config and string descriptors of the operations performed
    on the original config to produce the new config.

    Args:
        config: Original hyperparameter configuration.
        mutations: Specification of mutations to perform as documented
            in the PopulationBasedTraining scheduler.
        resample_probability: Probability of allowing resampling of a
            particular variable.
        perturbation_factors: Scaling factors to choose between when mutating
            a continuous hyperparameter.
        custom_explore_fn: Custom explore function applied after built-in
            config perturbations.

    Returns:
        new_config: New hyperparameter configuration (after random mutations).
        operations: Map of hyperparams -> strings describing mutation operations
            performed
    """
    operations = {}
    new_config = copy.deepcopy(config)
    for key, distribution in mutations.items():
        if isinstance(distribution, dict):
            nested_new_config, nested_ops = _explore(config[key], mutations[key], resample_probability, perturbation_factors, custom_explore_fn=None)
            new_config.update({key: nested_new_config})
            operations.update({key: nested_ops})
        elif isinstance(distribution, (list, tuple)):
            if random.random() < resample_probability or config[key] not in distribution:
                new_config[key] = random.choice(distribution)
                operations[key] = 'resample'
            else:
                shift = random.choice([-1, 1])
                old_idx = distribution.index(config[key])
                new_idx = old_idx + shift
                new_idx = min(max(new_idx, 0), len(distribution) - 1)
                new_config[key] = distribution[new_idx]
                operations[key] = f'shift {('left' if shift == -1 else 'right')}{(' (noop)' if old_idx == new_idx else '')}'
        elif isinstance(distribution, (Domain, Callable)):
            if random.random() < resample_probability:
                new_config[key] = distribution.sample(None) if isinstance(distribution, Domain) else distribution()
                operations[key] = 'resample'
            else:
                perturbation_factor = random.choice(perturbation_factors)
                new_config[key] = config[key] * perturbation_factor
                operations[key] = f'* {perturbation_factor}'
            if isinstance(config[key], int):
                new_config[key] = int(new_config[key])
        else:
            raise ValueError(f'Unsupported hyperparameter distribution type: {type(distribution)}')
    if custom_explore_fn:
        new_config = custom_explore_fn(new_config)
        assert new_config is not None, 'Custom explore fn failed to return new config'
    return (new_config, operations)