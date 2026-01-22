from typing import (
import numpy as np
import gymnasium as gym
from ray.rllib.utils.annotations import ExperimentalAPI
@ExperimentalAPI
class ActionConnectorDataType:
    """Data type that is fed into and yielded from agent connectors.

    Args:
        env_id: ID of the environment.
        agent_id: ID to help identify the agent from which the data is received.
        input_dict: Input data that was passed into the policy.
            Sometimes output must be adapted based on the input, for example
            action masking. So the entire input data structure is provided here.
        output: An object of PolicyOutputType. It is is composed of the
            action output, the internal state output, and additional data fetches.

    """

    def __init__(self, env_id: str, agent_id: str, input_dict: TensorStructType, output: PolicyOutputType):
        self.env_id = env_id
        self.agent_id = agent_id
        self.input_dict = input_dict
        self.output = output