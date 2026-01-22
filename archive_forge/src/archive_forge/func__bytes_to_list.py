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
def _bytes_to_list(content: bytes) -> List:
    """Parse bytes from a Response object into a Python list.

    Expects the response body to be JSON-encoded data.

    NOTE: This is exactly the same implementation as `_bytes_to_dict` and will not complain if the returned data is a
    dictionary. The only advantage of having both is to help the user (and mypy) understand what kind of data to expect.
    """
    return json.loads(content.decode())