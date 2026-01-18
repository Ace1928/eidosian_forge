from __future__ import annotations
from pathlib import Path
from typing import Callable
from gradio_client import file
from gradio_client.documentation import document
from gradio.components.base import Component
from gradio.data_classes import FileData
from gradio.events import Events

        Parameters:
            value: Expects function to return a {str} or {pathlib.Path} filepath of type (.obj, .glb, .stl, or .gltf)
        Returns:
            The uploaded file as an instance of `FileData`.
        