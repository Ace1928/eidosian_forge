import jinja2
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
import minerl.herobraine.hero.spaces as spaces
import numpy as np
class _IsAliveObservation(LifeStatsObservation):
    """
    Handles is_alive observation. Initial value is True (alive)
    """

    def __init__(self):
        keys = ['is_alive']
        super().__init__(hero_keys=keys, univ_keys=keys, space=spaces.Box(low=0, high=1, shape=(), dtype=bool), default_if_missing=1)