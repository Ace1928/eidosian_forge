from __future__ import annotations
import dataclasses
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast
import numpy as np
import PIL.Image
from gradio_client import file
from gradio_client.documentation import document
from typing_extensions import TypedDict
from gradio import image_utils, utils
from gradio.components.base import Component, server
from gradio.data_classes import FileData, GradioModel
from gradio.events import Events
class BlobData(TypedDict):
    type: str
    index: Optional[int]
    file: bytes
    id: str