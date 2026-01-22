from minerl.herobraine.hero import handlers
from typing import List
from minerl.herobraine.hero.handlers.translation import TranslationHandler
import time
from minerl.herobraine.env_specs.navigate_specs import Navigate
import coloredlogs
import logging
class NavigateWithDistanceMonitor(Navigate):

    def create_monitors(self) -> List[TranslationHandler]:
        return [handlers.CompassObservation(angle=False, distance=True)]