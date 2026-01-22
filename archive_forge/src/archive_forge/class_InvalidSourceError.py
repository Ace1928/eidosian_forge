from __future__ import annotations
import abc
import tempfile
from pathlib import Path, PurePath
from lazyops.utils.pooler import ThreadPoolV2
from lazyops.utils.logs import default_logger, null_logger, Logger
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar, TYPE_CHECKING
class InvalidSourceError(Exception):
    """
    Invalid Source Error
    """

    def __init__(self, converter: 'BaseConverter', source: str):
        """
        Initialize the Invalid Source Error
        """
        self.converter = converter
        self.source = source
        super().__init__(f'[{self.converter.name}] Invalid conversion source: {source} (supported: {self.converter.source})')