from __future__ import annotations
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, Literal
import gradio_client.utils as client_utils
from gradio_client import file
from gradio_client.documentation import document
from gradio import processing_utils
from gradio.components.base import Component
from gradio.data_classes import FileData, ListFiles
from gradio.events import Events
from gradio.utils import NamedString

        Parameters:
            value: Expects a `str` filepath or URL, or a `list[str]` of filepaths/URLs.
        Returns:
            File information as a FileData object, or a list of FileData objects.
        