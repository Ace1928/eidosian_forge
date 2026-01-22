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
def _dep_warn(field):
    warnings.warn(dedent(f'`{field}` kwargs of validate has been deprecated for security\n        reasons, and will be removed soon.\n\n        Please explicitly use the `n_changes, new_notebook = nbformat.validator.normalize(old_notebook, ...)` if you wish to\n        normalise your notebook. `normalize` is available since nbformat 5.5.0\n\n        '), DeprecationWarning, stacklevel=3)