import os
import logging
import warnings
import jinja2
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler
from minerl.herobraine.hero import spaces
from typing import Tuple
import numpy as np

        Combines two POV observations into one. If all of the properties match return self
        otherwise raise an exception.
        