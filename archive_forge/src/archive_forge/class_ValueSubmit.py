from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class ValueSubmit(ModelEvent):
    """ Announce a value being submitted on a text input widget.

    """
    event_name = 'value_submit'
    value: str

    def __init__(self, model: TextInput | None, value: str) -> None:
        from .models.widgets import TextInput
        if model is not None and (not isinstance(model, TextInput)):
            clsname = self.__class__.__name__
            raise ValueError(f'{clsname} event only applies to text input models')
        super().__init__(model=model)
        self.value = value