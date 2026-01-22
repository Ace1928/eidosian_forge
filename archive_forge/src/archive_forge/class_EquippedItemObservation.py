import json
from typing import List
import jinja2
from minerl.herobraine.hero.mc import EQUIPMENT_SLOTS
from minerl.herobraine.hero import spaces
from minerl.herobraine.hero.handlers.translation import TranslationHandler, TranslationHandlerGroup
import numpy as np
class EquippedItemObservation(TranslationHandlerGroup):
    """
    Enables the observation of equipped items in the main, offhand,
    and armor slots of the agent.

    """

    def to_string(self) -> str:
        return 'equipped_items'

    def xml_template(self) -> str:
        return str('<ObservationFromEquippedItem/>')

    def __init__(self, items: List[str], mainhand: bool=True, offhand: bool=False, armor: bool=False, _default: str='none', _other: str='other'):
        self.mainhand = mainhand
        self.offhand = offhand
        self.armor = armor
        self._items = items
        self._other = _other
        self._default = _default
        if self._other not in self._items:
            self._items.append(self._other)
        if self._default not in self._items:
            self._items.append(self._default)
        handlers = []
        if mainhand:
            handlers.append(_EquippedItemObservation(['mainhand'], self._items, _default=_default, _other=_other))
        if offhand:
            handlers.append(_EquippedItemObservation(['offhand'], self._items, _default=_default, _other=_other))
        if armor:
            handlers.extend([_EquippedItemObservation([slot], self._items, _default=_default, _other=_other) for slot in EQUIPMENT_SLOTS if slot not in ['mainhand', 'offhand']])
        super().__init__(handlers)

    def __eq__(self, other):
        return super().__eq__(other) and other.mainhand == self.mainhand and (other.offhand == self.offhand) and (other.armor == self.armor)

    def __or__(self, other):
        return EquippedItemObservation(items=list(set(self._items) | set(other._items)), mainhand=self.mainhand or other.mainhand, offhand=self.offhand or other.offhand, armor=self.armor or other.armor, _other=self._other, _default=self._default)