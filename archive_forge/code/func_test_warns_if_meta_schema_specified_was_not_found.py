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
def test_warns_if_meta_schema_specified_was_not_found(self):
    with self.assertWarns(DeprecationWarning) as cm:
        validators.validator_for(schema={'$schema': 'unknownSchema'})
    self.assertEqual(cm.filename, __file__)
    self.assertEqual(str(cm.warning), 'The metaschema specified by $schema was not found. Using the latest draft to validate, but this will raise an error in the future.')