import json
from typing import List
import jinja2
from minerl.herobraine.hero.mc import EQUIPMENT_SLOTS
from minerl.herobraine.hero import spaces
from minerl.herobraine.hero.handlers.translation import TranslationHandler, TranslationHandlerGroup
import numpy as np

        Initializes the space of the handler with a spaces.Dict
        of all of the spaces for each individual command.
        