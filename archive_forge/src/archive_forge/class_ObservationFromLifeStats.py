import jinja2
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
import minerl.herobraine.hero.mc as mc
import minerl.herobraine.hero.spaces as spaces
import numpy as np
class ObservationFromLifeStats(TranslationHandlerGroup):
    """Groups all of the lifestats observations together to correspond to one XML element.."""

    def to_string(self) -> str:
        return 'life_stats'

    def __init__(self):
        super(ObservationFromLifeStats, self).__init__(handlers=[_IsAliveObservation(), _LifeObservation(), _ScoreObservation(), _FoodObservation(), _SaturationObservation(), _XPObservation(), _BreathObservation()])

    def xml_template(self) -> str:
        return str('<ObservationFromFullStats/>')