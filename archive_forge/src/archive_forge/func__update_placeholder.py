from __future__ import annotations
import asyncio
import traceback
from enum import Enum
from inspect import (
from io import BytesIO
from typing import (
import param
from .._param import Margin
from ..io.resources import CDN_DIST
from ..layout import Feed, ListPanel
from ..layout.card import Card
from ..layout.spacer import VSpacer
from ..pane.image import SVG
from .message import ChatMessage
@param.depends('placeholder_text', 'placeholder_params', watch=True, on_init=True)
def _update_placeholder(self):
    loading_avatar = SVG(PLACEHOLDER_SVG, sizing_mode='fixed', width=35, height=35, css_classes=['rotating-placeholder'])
    self._placeholder = ChatMessage(self.placeholder_text, avatar=loading_avatar, css_classes=['message'], **self.placeholder_params)