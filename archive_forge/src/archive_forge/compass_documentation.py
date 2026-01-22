import jinja2
import numpy as np
from minerl.herobraine.hero import spaces
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
Initializes a compass observation. Forms

        Args:
            angle (bool, optional): Whether or not to include angle observation. Defaults to True.
            distance (bool, optional): Whether or not ot include distance observation. Defaults to False.
        