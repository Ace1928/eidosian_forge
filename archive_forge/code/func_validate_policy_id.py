import gymnasium as gym
import logging
import numpy as np
import re
from typing import (
import tree  # pip install dm_tree
import ray.cloudpickle as pickle
from ray.rllib.models.preprocessors import ATARI_OBS_SHAPE
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import (
from ray.util import log_once
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
def validate_policy_id(policy_id: str, error: bool=False) -> None:
    """Makes sure the given `policy_id` is valid.

    Args:
        policy_id: The Policy ID to check.
            IMPORTANT: Must not contain characters that
            are also not allowed in Unix/Win filesystems, such as: `<>:"/\\|?*`
            or a dot `.` or space ` ` at the end of the ID.
        error: Whether to raise an error (ValueError) or a warning in case of an
            invalid `policy_id`.

    Raises:
        ValueError: If the given `policy_id` is not a valid one and `error` is True.
    """
    if not isinstance(policy_id, str) or len(policy_id) == 0 or re.search('[<>:"/\\\\|?]', policy_id) or (policy_id[-1] in (' ', '.')):
        msg = f'PolicyID `{policy_id}` not valid! IDs must be a non-empty string, must not contain characters that are also disallowed file- or directory names on Unix/Windows and must not end with a dot `.` or a space ` `.'
        if error:
            raise ValueError(msg)
        elif log_once('invalid_policy_id'):
            logger.warning(msg)