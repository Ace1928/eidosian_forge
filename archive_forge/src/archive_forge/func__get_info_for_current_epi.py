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
def _get_info_for_current_epi(self, epi_is_done):
    if epi_is_done and self.output_additional_info:
        info_for_current_epi = self._get_episode_info()
    else:
        info_for_current_epi = None
    return info_for_current_epi