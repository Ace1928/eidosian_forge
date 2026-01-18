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
def test_unevaluated_properties_invalid_against_subschema(self):
    schema = {'properties': {'foo': {'type': 'string'}}, 'unevaluatedProperties': {'const': 12}}
    message = self.message_for(instance={'foo': 'foo', 'bar': 'bar', 'baz': 12}, schema=schema)
    self.assertEqual(message, "Unevaluated properties are not valid under the given schema ('bar' was unevaluated and invalid)")