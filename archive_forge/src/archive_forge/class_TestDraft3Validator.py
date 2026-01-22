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
class TestDraft3Validator(AntiDraft6LeakMixin, ValidatorTestMixin, TestCase):
    Validator = validators.Draft3Validator
    valid: tuple[dict, dict] = ({}, {})
    invalid = ({'type': 'integer'}, 'foo')

    def test_any_type_is_valid_for_type_any(self):
        validator = self.Validator({'type': 'any'})
        validator.validate(object())

    def test_any_type_is_redefinable(self):
        """
        Sigh, because why not.
        """
        Crazy = validators.extend(self.Validator, type_checker=self.Validator.TYPE_CHECKER.redefine('any', lambda checker, thing: isinstance(thing, int)))
        validator = Crazy({'type': 'any'})
        validator.validate(12)
        with self.assertRaises(exceptions.ValidationError):
            validator.validate('foo')

    def test_is_type_is_true_for_any_type(self):
        self.assertTrue(self.Validator({'type': 'any'}).is_valid(object()))

    def test_is_type_does_not_evade_bool_if_it_is_being_tested(self):
        self.assertTrue(self.Validator({}).is_type(True, 'boolean'))
        self.assertTrue(self.Validator({'type': 'any'}).is_valid(True))