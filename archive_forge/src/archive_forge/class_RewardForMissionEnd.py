import abc
from minerl.herobraine.hero.mc import strip_item_prefix
from minerl.herobraine.hero.spaces import Box
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler
import jinja2
from typing import List, Dict, Union
import numpy as np
class RewardForMissionEnd(RewardHandler):

    def to_string(self) -> str:
        return 'reward_for_mission_end'

    def xml_element(self) -> str:
        return str('<RewardForMissionEnd>\n                    <Reward description="{{ description }}" reward="{{ reward }}" />\n                </RewardForMissionEnd>')

    def __init__(self, reward: int, description: str='out_of_time'):
        """Creates a reward which is awarded when a mission ends."""
        super().__init__()
        self.reward = reward
        self.description = description

    def from_universal(self, obs):
        return 0