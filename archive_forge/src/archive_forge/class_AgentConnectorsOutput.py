from typing import (
import numpy as np
import gymnasium as gym
from ray.rllib.utils.annotations import ExperimentalAPI
@ExperimentalAPI
class AgentConnectorsOutput:
    """Final output data type of agent connectors.

    Args are populated depending on the AgentConnector settings.
    The branching happens in ViewRequirementAgentConnector.

    Args:
        raw_dict: The raw input dictionary that sampler can use to
            build episodes and training batches.
            This raw dict also gets passed into ActionConnectors in case
            it contains data useful for action adaptation (e.g. action masks).
        sample_batch: The SampleBatch that can be immediately used for
            querying the policy for next action.
    """

    def __init__(self, raw_dict: Dict[str, TensorStructType], sample_batch: 'SampleBatch'):
        self.raw_dict = raw_dict
        self.sample_batch = sample_batch