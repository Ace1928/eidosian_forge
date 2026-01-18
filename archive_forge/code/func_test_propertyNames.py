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
def test_propertyNames(self):
    instance = {'foo': 12}
    schema = {'propertyNames': {'not': {'const': 'foo'}}}
    validator = validators.Draft7Validator(schema)
    error, = validator.iter_errors(instance)
    self.assertEqual(error.validator, 'not')
    self.assertEqual(error.message, "'foo' should not be valid under {'const': 'foo'}")
    self.assertEqual(error.path, deque([]))
    self.assertEqual(error.json_path, '$')
    self.assertEqual(error.schema_path, deque(['propertyNames', 'not']))