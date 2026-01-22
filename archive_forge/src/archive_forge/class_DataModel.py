from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from .model import Model
@abstract
class DataModel(Model):
    __data_model__ = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)