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
def validation_success(self, instance_path):
    return ''