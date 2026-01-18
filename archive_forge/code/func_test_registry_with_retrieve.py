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
def test_registry_with_retrieve(self):

    def retrieve(uri):
        return DRAFT202012.create_resource({'type': 'integer'})
    registry = referencing.Registry(retrieve=retrieve)
    schema = {'$ref': 'https://example.com/'}
    validator = validators.Draft202012Validator(schema, registry=registry)
    self.assertEqual((validator.is_valid(12), validator.is_valid('foo')), (True, False))