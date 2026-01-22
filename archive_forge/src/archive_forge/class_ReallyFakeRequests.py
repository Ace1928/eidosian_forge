from __future__ import annotations
from collections import deque, namedtuple
from contextlib import contextmanager
from decimal import Decimal
from io import BytesIO
from typing import Any
from unittest import TestCase, mock
from urllib.request import pathname2url
import json
import os
import sys
import tempfile
import warnings
from attrs import define, field
from referencing.jsonschema import DRAFT202012
import referencing.exceptions
from jsonschema import (
@define
class ReallyFakeRequests:
    _responses: dict[str, Any]

    def get(self, url):
        response = self._responses.get(url)
        if url is None:
            raise ValueError('Unknown URL: ' + repr(url))
        return _ReallyFakeJSONResponse(json.dumps(response))