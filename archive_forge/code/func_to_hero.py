import jinja2
from typing import List
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
from minerl.herobraine.hero import spaces
import numpy as np
def to_hero(self, x):
    for key in self.hero_keys:
        x = x[key]
    return np.eye(2)[x]