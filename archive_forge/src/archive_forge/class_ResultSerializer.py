from __future__ import annotations
import itertools
import types
import warnings
from io import BytesIO
from typing import (
from urllib.parse import urlparse
from urllib.request import url2pathname
class ResultSerializer:

    def __init__(self, result: Result):
        self.result = result

    def serialize(self, stream: IO, encoding: str='utf-8', **kwargs: Any) -> None:
        """return a string properly serialized"""
        pass