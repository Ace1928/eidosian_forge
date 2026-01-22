import base64
import io
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
from requests import HTTPError
from ..constants import ENDPOINT
from ..utils import (
from ._text_generation import TextGenerationStreamResponse, _parse_text_generation_error
@dataclass
class ModelStatus:
    """
    This Dataclass represents the the model status in the Hugging Face Inference API.

    Args:
        loaded (`bool`):
            If the model is currently loaded into Hugging Face's InferenceAPI. Models
            are loaded on-demand, leading to the user's first request taking longer.
            If a model is loaded, you can be assured that it is in a healthy state.
        state (`str`):
            The current state of the model. This can be 'Loaded', 'Loadable', 'TooBig'.
            If a model's state is 'Loadable', it's not too big and has a supported
            backend. Loadable models are automatically loaded when the user first
            requests inference on the endpoint. This means it is transparent for the
            user to load a model, except that the first call takes longer to complete.
        compute_type (`Dict`):
            Information about the compute resource the model is using or will use, such as 'gpu' type and number of
            replicas.
        framework (`str`):
            The name of the framework that the model was built with, such as 'transformers'
            or 'text-generation-inference'.
    """
    loaded: bool
    state: str
    compute_type: Dict
    framework: str