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
class IteratedAsymPrisonersDilemma(TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma):
    """
    A two-agent environment for the Asymmetric Prisoner's Dilemma game.
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOUT_MATRIX = np.array([[[+0, -1], [-3, +0]], [[+0, -3], [-2, -2]]])
    NAME = 'IPD'