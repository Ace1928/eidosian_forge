import abc
from minerl.herobraine.hero.mc import strip_item_prefix
from minerl.herobraine.hero.spaces import Box
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler
import jinja2
from typing import List, Dict, Union
import numpy as np
class RewardForCollectingItemsOnce(_RewardForPosessingItemBase):
    """
    The standard malmo reward for collecting item once.

        rc = handlers.RewardForCollectingItemsOnce([
            dict(type="log", amount=1, reward=1.0),
        ])
    """

    def __init__(self, item_rewards: List[Dict[str, Union[str, int]]]):
        super().__init__(sparse=True, exclude_loops=True, item_rewards=item_rewards)
        self.seen_dict = dict()

    def from_universal(self, x):
        total_reward = 0
        if 'diff' in x and 'changes' in x['diff']:
            for change_json in x['diff']['changes']:
                item_name = strip_item_prefix(change_json['item'])
                if item_name == 'log2':
                    item_name = 'log'
                if item_name in self.reward_dict and 'quantity_change' in change_json and (item_name not in self.seen_dict):
                    if change_json['quantity_change'] > 0:
                        total_reward += self.reward_dict[item_name]['reward']
                        self.seen_dict[item_name] = True
        return total_reward