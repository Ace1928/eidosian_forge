import jinja2
from typing import List
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
from minerl.herobraine.hero import spaces
import numpy as np
class _FullStatsObservation(KeymapTranslationHandler):

    def to_hero(self, x) -> int:
        for key in self.hero_keys:
            x = x[key]
        return x

    def __init__(self, key_list: List[str], space=None, default_if_missing=None):
        if space is None:
            if 'achievement' == key_list[0]:
                space = spaces.Box(low=0, high=1, shape=(), dtype=int)
            else:
                space = spaces.Box(low=0, high=np.inf, shape=(), dtype=int)
        if default_if_missing is None:
            default_if_missing = np.zeros((), dtype=float)
        super().__init__(hero_keys=key_list, univ_keys=key_list, space=space, default_if_missing=default_if_missing)

    def xml_template(self) -> str:
        return str('<ObservationFromFullStats/>')