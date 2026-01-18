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

        Parameters:
            value: Expects a {dict} with "text" and "files", both optional. The files array is a list of file paths or URLs.
        Returns:
            The value to display in the multimodal textbox. Files information as a list of FileData objects.
        