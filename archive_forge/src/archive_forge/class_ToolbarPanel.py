from __future__ import annotations
import logging # isort:skip
from ....core.properties import Instance
from .html_annotation import HTMLAnnotation
class ToolbarPanel(HTMLAnnotation):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    toolbar = Instance('.models.tools.Toolbar', help='\n    A toolbar to display.\n    ')