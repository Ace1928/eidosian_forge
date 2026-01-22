from collections import OrderedDict
import logging
import numpy as np
from minerl.herobraine.hero.spaces import MineRLSpace
import minerl.herobraine.hero.spaces as spaces
from typing import List, Any
import typing
from minerl.herobraine.hero.handler import Handler
class KeymapTranslationHandler(TranslationHandler):

    def __init__(self, hero_keys: typing.List[str], univ_keys: typing.List[str], space: MineRLSpace, default_if_missing=None, to_string: str=None):
        """
        Wrapper for simple observations which just remaps keys.
        :param keys: list of nested dictionary keys from the root of the observation dict
        :param space: gym space corresponding to the shape of the returned value
        :param default_if_missing: value for handler to take if missing in the observation dict
        """
        super().__init__(space)
        self._to_string = to_string if to_string else hero_keys[-1]
        self.hero_keys = hero_keys
        self.univ_keys = univ_keys
        self.default_if_missing = default_if_missing
        self.logger = logging.getLogger(f'{__name__}.{self.to_string()}')

    def walk_dict(self, d, keys):
        for key in keys:
            if key in d:
                d = d[key]
            elif self.default_if_missing is not None:
                return np.array(self.default_if_missing)
            else:
                raise KeyError()
        return np.array(d)

    def to_hero(self, x) -> str:
        """What does it mean to do a keymap translation here?
        Since hero sends things as commands perhaps we could generalize
        this beyond observations.
        """
        raise NotImplementedError()

    def from_hero(self, hero_dict):
        return self.walk_dict(hero_dict, self.hero_keys)

    def from_universal(self, univ_dict):
        return self.walk_dict(univ_dict, self.univ_keys)

    def to_string(self) -> str:
        return self._to_string