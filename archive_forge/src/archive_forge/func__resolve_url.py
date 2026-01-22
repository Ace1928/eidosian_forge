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
def _resolve_url(self, model: Optional[str]=None, task: Optional[str]=None) -> str:
    model = model or self.model
    if model is not None and (model.startswith('http://') or model.startswith('https://')):
        return model
    if model is None:
        if task is None:
            raise ValueError('You must specify at least a model (repo_id or URL) or a task, either when instantiating `InferenceClient` or when making a request.')
        model = self.get_recommended_model(task)
        logger.info(f"Using recommended model {model} for task {task}. Note that it is encouraged to explicitly set `model='{model}'` as the recommended models list might get updated without prior notice.")
    return f'{INFERENCE_ENDPOINT}/pipeline/{task}/{model}' if task in ('feature-extraction', 'sentence-similarity') else f'{INFERENCE_ENDPOINT}/models/{model}'