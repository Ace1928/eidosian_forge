import logging
from functools import lru_cache
import os
import ray
import time
from typing import Dict, Optional, Tuple
from ray.tune.execution.cluster_info import _is_ray_cluster
from ray.tune.experiment import Trial
def _get_trial_cpu_and_gpu(trial: Trial) -> Tuple[int, int]:
    cpu = trial.placement_group_factory.required_resources.get('CPU', 0)
    gpu = trial.placement_group_factory.required_resources.get('GPU', 0)
    return (cpu, gpu)