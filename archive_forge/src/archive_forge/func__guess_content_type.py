import logging
import os
from collections.abc import Mapping
from email.headerregistry import Address
from functools import partial, reduce
from inspect import cleandoc
from itertools import chain
from types import MappingProxyType
from typing import (
from ..errors import RemovedConfigError
from ..warnings import SetuptoolsWarning
def _guess_content_type(file: str) -> Optional[str]:
    _, ext = os.path.splitext(file.lower())
    if not ext:
        return None
    if ext in _CONTENT_TYPES:
        return _CONTENT_TYPES[ext]
    valid = ', '.join((f'{k} ({v})' for k, v in _CONTENT_TYPES.items()))
    msg = f'only the following file extensions are recognized: {valid}.'
    raise ValueError(f'Undefined content type for {file}, {msg}')