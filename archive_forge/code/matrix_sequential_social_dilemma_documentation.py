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

        We want to be able to use a policy trained as player 1
        for evaluation as player 2 and vice versa.
        