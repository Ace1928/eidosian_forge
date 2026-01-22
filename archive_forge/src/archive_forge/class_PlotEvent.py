from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class PlotEvent(ModelEvent):
    """ The base class for all events applicable to Plot models.

    """

    def __init__(self, model: Plot | None) -> None:
        from .models import Plot
        if model is not None and (not isinstance(model, Plot)):
            raise ValueError(f'{self.__class__.__name__} event only applies to Plot models')
        super().__init__(model)