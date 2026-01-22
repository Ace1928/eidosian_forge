import logging
from functools import lru_cache
import os
import ray
import time
from typing import Dict, Optional, Tuple
from ray.tune.execution.cluster_info import _is_ray_cluster
from ray.tune.experiment import Trial
@lru_cache()
def _get_insufficient_resources_warning_msg(for_train: bool=False, trial: Optional[Trial]=None) -> str:
    msg = 'Ignore this message if the cluster is autoscaling. '
    if for_train:
        start = MSG_TRAIN_START
        insufficient = MSG_TRAIN_INSUFFICIENT
        end = MSG_TRAIN_END
    else:
        start = MSG_TUNE_START
        insufficient = MSG_TUNE_INSUFFICIENT
        end = MSG_TUNE_END
    msg += start.format(wait_time=_get_insufficient_resources_warning_threshold())
    if trial:
        asked_cpus, asked_gpus = _get_trial_cpu_and_gpu(trial)
        cluster_resources = _get_cluster_resources_no_autoscaler()
        msg += insufficient.format(asked_cpus=asked_cpus, asked_gpus=asked_gpus, cluster_cpus=cluster_resources.get('CPU', 0), cluster_gpus=cluster_resources.get('GPU', 0))
    msg += end
    return msg