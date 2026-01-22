import jinja2
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
import minerl.herobraine.hero.spaces as spaces
import numpy as np
class _LifeObservation(LifeStatsObservation):
    """
    Handles life observation / health observation. Its initial value on world creation is 20 (full bar)
    """

    def __init__(self):
        keys = ['life']
        super().__init__(hero_keys=keys, univ_keys=keys, space=spaces.Box(low=0, high=mc.MAX_LIFE, shape=(), dtype=float), default_if_missing=mc.MAX_LIFE)