from __future__ import annotations
import inspect
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Tuple, Union
from gradio_client import utils as client_utils
from gradio_client.documentation import document
from gradio import utils
from gradio.components.base import Component
from gradio.data_classes import FileData, GradioModel, GradioRootModel
from gradio.events import Events
class ChatbotData(GradioRootModel):
    root: List[Tuple[Union[str, FileMessage, None], Union[str, FileMessage, None]]]