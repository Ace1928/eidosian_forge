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
class MetaSchemaTestsMixin:

    def test_invalid_properties(self):
        with self.assertRaises(exceptions.SchemaError):
            self.Validator.check_schema({'properties': 12})

    def test_minItems_invalid_string(self):
        with self.assertRaises(exceptions.SchemaError):
            self.Validator.check_schema({'minItems': '1'})

    def test_enum_allows_empty_arrays(self):
        """
        Technically, all the spec says is they SHOULD have elements, not MUST.

        (As of Draft 6. Previous drafts do say MUST).

        See #529.
        """
        if self.Validator in {validators.Draft3Validator, validators.Draft4Validator}:
            with self.assertRaises(exceptions.SchemaError):
                self.Validator.check_schema({'enum': []})
        else:
            self.Validator.check_schema({'enum': []})

    def test_enum_allows_non_unique_items(self):
        """
        Technically, all the spec says is they SHOULD be unique, not MUST.

        (As of Draft 6. Previous drafts do say MUST).

        See #529.
        """
        if self.Validator in {validators.Draft3Validator, validators.Draft4Validator}:
            with self.assertRaises(exceptions.SchemaError):
                self.Validator.check_schema({'enum': [12, 12]})
        else:
            self.Validator.check_schema({'enum': [12, 12]})

    def test_schema_with_invalid_regex(self):
        with self.assertRaises(exceptions.SchemaError):
            self.Validator.check_schema({'pattern': '*notaregex'})

    def test_schema_with_invalid_regex_with_disabled_format_validation(self):
        self.Validator.check_schema({'pattern': '*notaregex'}, format_checker=None)