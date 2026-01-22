from __future__ import annotations
from typing import Any, List
import gradio_client.utils as client_utils
import numpy as np
import PIL.Image
from gradio_client import file
from gradio_client.documentation import document
from gradio import processing_utils, utils
from gradio.components.base import Component
from gradio.data_classes import FileData, GradioModel
from gradio.events import Events
class AnnotatedImageData(GradioModel):
    image: FileData
    annotations: List[Annotation]