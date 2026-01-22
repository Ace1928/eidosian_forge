import base64
import io
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
from requests import HTTPError
from huggingface_hub.errors import (
from ..constants import ENDPOINT
from ..utils import (
from ._generated.types import (
def _stream_chat_completion_response_from_bytes(bytes_lines: Iterable[bytes]) -> Iterable[ChatCompletionStreamOutput]:
    """Used in `InferenceClient.chat_completion` if model is served with TGI."""
    for item in bytes_lines:
        output = _format_chat_completion_stream_output_from_text_generation_from_bytes(item)
        if output is not None:
            yield output