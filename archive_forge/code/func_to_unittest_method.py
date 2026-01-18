from __future__ import annotations
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any
import json
import os
import re
import subprocess
import sys
import unittest
from attrs import field, frozen
from referencing import Registry
import referencing.jsonschema
from jsonschema.validators import _VALIDATORS
import jsonschema
def to_unittest_method(self, skip=lambda test: None, **kwargs):
    if self.valid:

        def fn(this):
            self.validate(**kwargs)
    else:

        def fn(this):
            with this.assertRaises(jsonschema.ValidationError):
                self.validate(**kwargs)
    fn.__name__ = '_'.join(['test', _DELIMITERS.sub('_', self.subject), _DELIMITERS.sub('_', self.case_description), _DELIMITERS.sub('_', self.description)])
    reason = skip(self)
    if reason is None or os.environ.get('JSON_SCHEMA_DEBUG', '0') != '0':
        return fn
    elif os.environ.get('JSON_SCHEMA_EXPECTED_FAILURES', '0') != '0':
        return unittest.expectedFailure(fn)
    else:
        return unittest.skip(reason)(fn)