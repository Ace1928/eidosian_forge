from __future__ import annotations
from typing import Any, Callable, List, Union
from gradio_client.documentation import document
from gradio.components.base import Component
from gradio.data_classes import GradioModel, GradioRootModel
from gradio.events import Events
class HighlightedTextData(GradioRootModel):
    root: List[HighlightedToken]