from __future__ import annotations
import warnings
from typing import (
import numpy as np
import semantic_version
from gradio_client.documentation import document
from gradio.components import Component
from gradio.data_classes import GradioModel
from gradio.events import Events
@staticmethod
def __get_cell_style(cell_id: str, cell_styles: list[dict]) -> str:
    styles_for_cell = []
    for style in cell_styles:
        if cell_id in style.get('selectors', []):
            styles_for_cell.extend(style.get('props', []))
    styles_str = '; '.join([f'{prop}: {value}' for prop, value in styles_for_cell])
    return styles_str