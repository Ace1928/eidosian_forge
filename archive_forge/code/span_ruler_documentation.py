import warnings
from functools import partial
from pathlib import Path
from typing import (
import srsly
from .. import util
from ..errors import Errors, Warnings
from ..language import Language
from ..matcher import Matcher, PhraseMatcher
from ..matcher.levenshtein import levenshtein_compare
from ..scorer import Scorer
from ..tokens import Doc, Span
from ..training import Example
from ..util import SimpleFrozenList, ensure_path, registry
from .pipe import Pipe
Save the span ruler patterns to a directory.

        path (Union[str, Path]): A path to a directory.

        DOCS: https://spacy.io/api/spanruler#to_disk
        