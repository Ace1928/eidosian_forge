import jinja2
import numpy as np
from minerl.herobraine.hero import spaces
from minerl.herobraine.hero.handlers.translation import KeymapTranslationHandler, TranslationHandlerGroup
class CompassObservation(TranslationHandlerGroup):

    def to_string(self) -> str:
        return 'compass'

    def xml_template(self) -> str:
        return str('<ObservationFromCompass/>')

    def __init__(self, angle=True, distance=False):
        """Initializes a compass observation. Forms

        Args:
            angle (bool, optional): Whether or not to include angle observation. Defaults to True.
            distance (bool, optional): Whether or not ot include distance observation. Defaults to False.
        """
        assert angle or distance, 'Must observe either angle or distance'
        handlers = []
        if angle:
            handlers.append(_CompassAngleObservation())
        if distance:
            handlers.append(KeymapTranslationHandler(hero_keys=['distanceToCompassTarget'], univ_keys=['compass', 'distance'], to_string='distance', space=spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32)))
        super(CompassObservation, self).__init__(handlers=handlers)