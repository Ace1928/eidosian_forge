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
def _get_errors(nbdict: Any, version: int, version_minor: int, relax_add_props: bool, *args: Any) -> Any:
    validator = get_validator(version, version_minor, relax_add_props=relax_add_props)
    if not validator:
        msg = f'No schema for validating v{version}.{version_minor} notebooks'
        raise ValidationError(msg)
    iter_errors = validator.iter_errors(nbdict, *args)
    errors = list(iter_errors)
    if len(errors) and validator.name != 'jsonschema':
        validator = get_validator(version=version, version_minor=version_minor, relax_add_props=relax_add_props, name='jsonschema')
        return validator.iter_errors(nbdict, *args)
    return iter(errors)