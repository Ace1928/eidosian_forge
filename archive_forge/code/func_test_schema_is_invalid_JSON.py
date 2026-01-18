from contextlib import redirect_stderr, redirect_stdout
from importlib import metadata
from io import StringIO
from json import JSONDecodeError
from pathlib import Path
from textwrap import dedent
from unittest import TestCase
import json
import os
import subprocess
import sys
import tempfile
import warnings
from jsonschema import Draft4Validator, Draft202012Validator
from jsonschema.exceptions import (
from jsonschema.validators import _LATEST_VERSION, validate
def test_schema_is_invalid_JSON(self):
    schema = 'not valid JSON!'
    self.assertOutputs(files=dict(some_schema=schema), argv=['some_schema'], exit_code=1, stderr=f"                Failed to parse 'some_schema': {_message_for(schema)}\n            ")