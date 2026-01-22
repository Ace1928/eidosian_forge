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
def _import_numpy():
    """Make sure `numpy` is installed on the machine."""
    if not is_numpy_available():
        raise ImportError('Please install numpy to use deal with embeddings (`pip install numpy`).')
    import numpy
    return numpy