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
def test_refresolver_with_pointer_in_schema_with_no_id(self):
    """
        See https://github.com/python-jsonschema/jsonschema/issues/1124#issuecomment-1632574249.
        """
    schema = {'properties': {'x': {'$ref': '#/definitions/x'}}, 'definitions': {'x': {'type': 'integer'}}}
    validator = validators.Draft202012Validator(schema, resolver=validators._RefResolver('', schema))
    self.assertEqual((validator.is_valid({'x': 'y'}), validator.is_valid({'x': 37})), (False, True))