import jinja2
from typing import List
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
from minerl.herobraine.hero import spaces
import numpy as np
class _CanSeeSkyObservation(_FullStatsObservation):

    def to_hero(self, x):
        for key in self.hero_keys:
            x = x[key]
        return np.eye(2)[x]

    def __init__(self):
        super().__init__(key_list=['can_see_sky'], space=spaces.Box(low=0, high=1, shape=(), dtype=int), default_if_missing=1)