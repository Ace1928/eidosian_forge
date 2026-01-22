import jinja2
from typing import List
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
from minerl.herobraine.hero import spaces
import numpy as np
class ObservationFromCurrentLocation(TranslationHandlerGroup):
    """
    Includes the current biome, how likely rain and snow are there, as well as the current light level, how bright the
    sky is, and if the player can see the sky.

    Also includes x, y, z, roll, and pitch
    """

    def xml_template(self) -> str:
        return str('<ObservationFromFullStats/>')

    def to_string(self) -> str:
        return 'location_stats'

    def __init__(self):
        super(ObservationFromCurrentLocation, self).__init__(handlers=[_SunBrightnessObservation(), _SkyLightLevelObservation(), _LightLevelObservation(), _CanSeeSkyObservation(), _BiomeRainfallObservation(), _BiomeTemperatureObservation(), _IsRainingObservation(), _BiomeIDObservation(), _PitchObservation(), _YawObservation(), _XPositionObservation(), _YPositionObservation(), _ZPositionObservation(), _SeaLevelObservation()])