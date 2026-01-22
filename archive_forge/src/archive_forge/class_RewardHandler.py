import abc
from minerl.herobraine.hero.mc import strip_item_prefix
from minerl.herobraine.hero.spaces import Box
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler
import jinja2
from typing import List, Dict, Union
import numpy as np
class RewardHandler(TranslationHandler):
    """
    Specifies a reward handler for a task.
    These need to be attached to tasks with reinforcement learning objectives.
    All rewards need inherit from this reward handler
    #Todo: Figure out how this interplays with Hero, as rewards are summed.
    """

    def __init__(self):
        super().__init__(Box(-np.inf, np.inf, shape=()))

    def from_hero(self, obs_dict):
        """
        By default hero will include the reward in the observation.
        This is just a pass through for convenience.
        :param obs_dict:
        :return: The reward
        """
        return obs_dict['reward']