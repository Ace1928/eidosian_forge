from __future__ import annotations
import tempfile
from pathlib import Path
from typing import Callable, Literal
from gradio_client import file
from gradio_client.documentation import document
from gradio.components.base import Component
from gradio.data_classes import FileData
from gradio.events import Events

        Parameters:
            value: Expects a `str` or `pathlib.Path` filepath
        Returns:
            File information as a FileData object
        