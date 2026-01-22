from __future__ import annotations
import json
import os
import secrets
import tempfile
import uuid
from pathlib import Path
from typing import Any
from gradio_client import media_data, utils
from gradio_client.data_classes import FileData
def _serialize_single(self, x: str | FileData | None, load_dir: str | Path='', allow_links: bool=False) -> FileData | None:
    if x is None or isinstance(x, dict):
        return x
    if utils.is_http_url_like(x):
        filename = x
        size = None
    else:
        filename = str(Path(load_dir) / x)
        size = Path(filename).stat().st_size
    return {'name': filename or None, 'data': None if allow_links else utils.encode_url_or_file_to_base64(filename), 'orig_name': Path(filename).name, 'size': size}