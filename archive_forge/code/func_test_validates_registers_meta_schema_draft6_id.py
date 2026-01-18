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
def test_validates_registers_meta_schema_draft6_id(self):
    meta_schema_key = 'meta schema $id'
    my_meta_schema = {'$id': meta_schema_key}
    validators.create(meta_schema=my_meta_schema, version='my version')
    self.addCleanup(validators._META_SCHEMAS.pop, meta_schema_key)
    self.addCleanup(validators._VALIDATORS.pop, 'my version')
    self.assertIn(meta_schema_key, validators._META_SCHEMAS)