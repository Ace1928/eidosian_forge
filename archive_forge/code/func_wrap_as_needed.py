import collections
from functools import partial
import itertools
import sys
from numbers import Number
from typing import Dict, Iterator, Set, Union
from typing import List, Optional
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.utils.annotations import DeveloperAPI, ExperimentalAPI, PublicAPI
from ray.rllib.utils.compression import pack, unpack, is_compressed
from ray.rllib.utils.deprecation import Deprecated, deprecation_warning
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import (
from ray.util import log_once
@staticmethod
@PublicAPI
def wrap_as_needed(policy_batches: Dict[PolicyID, SampleBatch], env_steps: int) -> Union[SampleBatch, 'MultiAgentBatch']:
    """Returns SampleBatch or MultiAgentBatch, depending on given policies.
        If policy_batches is empty (i.e. {}) it returns an empty MultiAgentBatch.

        Args:
            policy_batches: Mapping from policy ids to SampleBatch.
            env_steps: Number of env steps in the batch.

        Returns:
            The single default policy's SampleBatch or a MultiAgentBatch
            (more than one policy).
        """
    if len(policy_batches) == 1 and DEFAULT_POLICY_ID in policy_batches:
        return policy_batches[DEFAULT_POLICY_ID]
    return MultiAgentBatch(policy_batches=policy_batches, env_steps=env_steps)