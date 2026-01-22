import asyncio
import base64
import logging
import time
import warnings
from dataclasses import asdict
from typing import (
from requests.structures import CaseInsensitiveDict
from huggingface_hub.constants import ALL_INFERENCE_API_FRAMEWORKS, INFERENCE_ENDPOINT, MAIN_INFERENCE_API_FRAMEWORKS
from huggingface_hub.inference._common import (
from huggingface_hub.inference._text_generation import (
from huggingface_hub.inference._types import (
from huggingface_hub.utils import (
from .._common import _async_yield_from, _import_aiohttp
def _unpack_response(framework: str, items: List[Dict]) -> None:
    for model in items:
        if framework == 'sentence-transformers':
            models_by_task.setdefault('feature-extraction', []).append(model['model_id'])
            models_by_task.setdefault('sentence-similarity', []).append(model['model_id'])
        else:
            models_by_task.setdefault(model['task'], []).append(model['model_id'])