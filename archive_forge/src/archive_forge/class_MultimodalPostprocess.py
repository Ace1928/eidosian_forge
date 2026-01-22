from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, List, Literal, TypedDict
import gradio_client.utils as client_utils
from gradio_client.documentation import document
from pydantic import Field
from typing_extensions import NotRequired
from gradio.components.base import FormComponent
from gradio.data_classes import FileData, GradioModel
from gradio.events import Events
class MultimodalPostprocess(TypedDict):
    text: str
    files: List[FileData]