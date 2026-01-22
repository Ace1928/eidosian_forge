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
def _b64_encode(content: ContentT) -> str:
    """Encode a raw file (image, audio) into base64. Can be byes, an opened file, a path or a URL."""
    with _open_as_binary(content) as data:
        data_as_bytes = data if isinstance(data, bytes) else data.read()
        return base64.b64encode(data_as_bytes).decode()