from __future__ import annotations
import pathlib
from typing import TYPE_CHECKING, Literal
import param
from ...config import config
from ...io.resources import JS_URLS
from ..base import BasicTemplate
def _apply_root(self, name, model, tags):
    if 'main' in tags:
        model.margin = (10, 15, 10, 10)