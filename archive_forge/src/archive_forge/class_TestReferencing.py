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
class TestReferencing(TestCase):

    def test_registry_with_retrieve(self):

        def retrieve(uri):
            return DRAFT202012.create_resource({'type': 'integer'})
        registry = referencing.Registry(retrieve=retrieve)
        schema = {'$ref': 'https://example.com/'}
        validator = validators.Draft202012Validator(schema, registry=registry)
        self.assertEqual((validator.is_valid(12), validator.is_valid('foo')), (True, False))

    def test_custom_registries_do_not_autoretrieve_remote_resources(self):
        registry = referencing.Registry()
        schema = {'$ref': 'https://example.com/'}
        validator = validators.Draft202012Validator(schema, registry=registry)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            with self.assertRaises(referencing.exceptions.Unresolvable):
                validator.validate(12)
        self.assertFalse(w)