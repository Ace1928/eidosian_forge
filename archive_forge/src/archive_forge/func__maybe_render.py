from collections import defaultdict
import logging
import time
import tree  # pip install dm_tree
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, Tuple, Union
import numpy as np
from ray.rllib.env.base_env import ASYNC_RESET_RETURN, BaseEnv
from ray.rllib.env.external_env import ExternalEnvWrapper
from ray.rllib.env.wrappers.atari_wrappers import MonitorEnv, get_wrapper_by_cls
from ray.rllib.evaluation.collectors.simple_list_collector import _PolicyCollectorGroup
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch, concat_samples
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.filter import Filter
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import unbatch, get_original_space
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
def _maybe_render(self):
    """Visualize environment."""
    if not self._render or not self._simple_image_viewer:
        return
    t5 = time.time()
    rendered = self._base_env.try_render()
    if isinstance(rendered, np.ndarray) and len(rendered.shape) == 3:
        self._simple_image_viewer.imshow(rendered)
    elif rendered not in [True, False, None]:
        raise ValueError(f"The env's ({self._base_env}) `try_render()` method returned an unsupported value! Make sure you either return a uint8/w x h x 3 (RGB) image or handle rendering in a window and then return `True`.")
    self._perf_stats.incr('env_render_time', time.time() - t5)