from __future__ import annotations
import logging # isort:skip
from datetime import datetime
from typing import (
from .core.serialization import Deserializer
class MenuItemClick(ModelEvent):
    """ Announce a button click event on a Bokeh menu item.

    """
    event_name = 'menu_item_click'

    def __init__(self, model: Model, item: str | None=None) -> None:
        self.item = item
        super().__init__(model=model)