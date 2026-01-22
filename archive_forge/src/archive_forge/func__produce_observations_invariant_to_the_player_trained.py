import logging
from abc import ABC
from collections import Iterable
from typing import Dict, Optional
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.examples.env.utils.interfaces import InfoAccumulationInterface
from ray.rllib.examples.env.utils.mixins import (
def _produce_observations_invariant_to_the_player_trained(self, action_player_0: int, action_player_1: int):
    """
        We want to be able to use a policy trained as player 1
        for evaluation as player 2 and vice versa.
        """
    return [action_player_0 * self.NUM_ACTIONS + action_player_1, action_player_1 * self.NUM_ACTIONS + action_player_0]