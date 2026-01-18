import logging
import threading
import time
from typing import Union, Optional
from enum import Enum
import ray.cloudpickle as pickle
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import (
@PublicAPI
def update_policy_weights(self) -> None:
    """Query the server for new policy weights, if local inference is enabled."""
    self._update_local_policy(force=True)