from importlib import metadata
from json import JSONDecodeError
from textwrap import dedent
import argparse
import json
import sys
import traceback
import warnings
from attrs import define, field
from jsonschema.exceptions import SchemaError
from jsonschema.validators import _RefResolver, validator_for
@define
class _PlainFormatter:
    _error_format = field()

    def filenotfound_error(self, path, exc_info):
        return f'{path!r} does not exist.\n'

    def parsing_error(self, path, exc_info):
        return 'Failed to parse {}: {}\n'.format('<stdin>' if path == '<stdin>' else repr(path), exc_info[1])

    def validation_error(self, instance_path, error):
        return self._error_format.format(file_name=instance_path, error=error)

    def validation_success(self, instance_path):
        return ''