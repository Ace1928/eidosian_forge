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
def __validate_headers(headers: list[str] | None, col_count: int):
    if headers is not None and len(headers) != col_count:
        raise ValueError(f'The length of the headers list must be equal to the col_count int.\nThe column count is set to {col_count} but `headers` has {len(headers)} items. Check the values passed to `col_count` and `headers`.')