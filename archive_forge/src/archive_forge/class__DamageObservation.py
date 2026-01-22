import json
from typing import List
import jinja2
from minerl.herobraine.hero.mc import EQUIPMENT_SLOTS
from minerl.herobraine.hero import spaces
from minerl.herobraine.hero.handlers.translation import TranslationHandler, TranslationHandlerGroup
import numpy as np
class _DamageObservation(TranslationHandler):
    """
    Returns a damage observation from a type str.
    """

    def __init__(self, keys: List[str], type_str: str):
        """
        Initializes the space of the handler with a spaces.Dict
        of all of the spaces for each individual command.
        """
        self._keys = keys
        self.type_str = type_str
        self._default = 0
        super().__init__(spaces.Box(low=-1, high=1562, shape=(), dtype=int))

    def to_string(self):
        return self.type_str

    def from_hero(self, info):
        try:
            head = info['equipped_items']
            for key in self._keys:
                head = json.loads(head[key])
            return np.array(head[self.type_str])
        except KeyError:
            return np.array(self._default, dtype=self.space.dtype)

    def from_universal(self, obs):
        try:
            if self._keys[0] == 'mainhand' and len(self._keys) == 1:
                offset = -9
                hotbar_index = obs['hotbar']
                if obs['slots']['gui']['type'] == 'class net.minecraft.inventory.ContainerPlayer':
                    offset -= 1
                return np.array(obs['slots']['gui']['slots'][offset + hotbar_index][self.type_str], dtype=np.int32)
            else:
                raise NotImplementedError('damage not implemented for hand type' + str(self._keys))
        except KeyError:
            return np.array(self._default, dtype=np.int32)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._keys == other._keys and (self.type_str == other.type_str)