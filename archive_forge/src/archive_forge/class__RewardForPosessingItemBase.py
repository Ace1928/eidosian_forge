import abc
from minerl.herobraine.hero.mc import strip_item_prefix
from minerl.herobraine.hero.spaces import Box
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler
import jinja2
from typing import List, Dict, Union
import numpy as np
class _RewardForPosessingItemBase(RewardHandler):

    def to_string(self) -> str:
        return 'reward_for_posessing_item'

    def xml_template(self) -> str:
        return str('<RewardForPossessingItem sparse="{{ sparse | lower }}" excludeLoops="{{ exclude_loops | string | lower}}">\n                    {% for item in items %}\n                    <Item amount="{{ item.amount }}" reward="{{ item.reward }}" type="{{ item.type }}" />\n                    {% endfor %}\n                </RewardForPossessingItem>\n                ')

    def __init__(self, sparse: bool, exclude_loops: bool, item_rewards: List[Dict[str, Union[str, int]]]):
        """Creates a reward which gives rewards based on items in the 
        inventory that are provided.

        See Malmo for documentation.
        """
        super().__init__()
        self.sparse = sparse
        self.exclude_loops = exclude_loops
        self.items = item_rewards
        self.reward_dict = {a['type']: dict(reward=a['reward'], amount=a['amount']) for a in self.items}
        for k, v in self.reward_dict.items():
            assert int(v['amount']) <= 1, 'Currently from universal is not implemented for item amounts > 1'
        for item in self.items:
            assert set(item.keys()) == {'amount', 'reward', 'type'}

    @abc.abstractmethod
    def from_universal(self, obs):
        raise NotImplementedError()