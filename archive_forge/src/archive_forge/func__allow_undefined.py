from __future__ import annotations
import json
import pprint
import warnings
from copy import deepcopy
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional
from ._imports import import_item
from .corpus.words import generate_corpus_id
from .json_compat import ValidationError, _validator_for_name, get_current_validator
from .reader import get_version
from .warnings import DuplicateCellId, MissingIDFieldWarning
def _allow_undefined(schema):
    schema['definitions']['cell']['oneOf'].append({'$ref': '#/definitions/unrecognized_cell'})
    schema['definitions']['output']['oneOf'].append({'$ref': '#/definitions/unrecognized_output'})
    return schema