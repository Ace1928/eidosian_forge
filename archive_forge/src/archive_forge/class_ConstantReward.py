import abc
from minerl.herobraine.hero.mc import strip_item_prefix
from minerl.herobraine.hero.spaces import Box
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler
import jinja2
from typing import List, Dict, Union
import numpy as np
class ConstantReward(RewardHandler):
    """
    A constant reward handler
    """

    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def from_hero(self, obs_dict):
        return self.constant

    def from_universal(self, x):
        return self.constant