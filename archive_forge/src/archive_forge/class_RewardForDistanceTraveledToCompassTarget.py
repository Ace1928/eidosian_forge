import abc
from minerl.herobraine.hero.mc import strip_item_prefix
from minerl.herobraine.hero.spaces import Box
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler
import jinja2
from typing import List, Dict, Union
import numpy as np
class RewardForDistanceTraveledToCompassTarget(RewardHandler):

    def to_string(self) -> str:
        return 'reward_for_distance_traveled_to_compass_target'

    def xml_template(self) -> str:
        return str('<RewardForDistanceTraveledToCompassTarget rewardPerBlock="{{ reward_per_block }}" density="{{ density }}"/>')

    def __init__(self, reward_per_block: int, density: str='PER_TICK'):
        """Creates a reward which is awarded when the player reaches a certain distance from a target."""
        self.reward_per_block = reward_per_block
        self.density = density
        self._prev_delta = None

    def from_universal(self, obs):
        if 'compass' in obs and 'deltaDistance' in obs['compass']:
            try:
                target = obs['compass']['target']
                target_pos = np.array([target['x'], target['y'], target['z']])
                position = obs['compass']['position']
                cur_pos = np.array([position['x'], position['y'], position['z']])
                delta = np.linalg.norm(target_pos - cur_pos)
                if not self._prev_delta:
                    return 0
                else:
                    return self._prev_delta - delta
            finally:
                self._prev_delta = delta

    def reset(self):
        self._prev_delta = None