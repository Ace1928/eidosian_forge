import itertools
from pathlib import Path
from typing import Any, Dict, Optional
import typer
from thinc.api import (
from wasabi import msg
from spacy.training import Example
from spacy.util import resolve_dot_names
from .. import util
from ..schemas import ConfigSchemaTraining
from ..util import registry
from ._util import (
def _sentences():
    return ['Apple is looking at buying U.K. startup for $1 billion', 'Autonomous cars shift insurance liability toward manufacturers', 'San Francisco considers banning sidewalk delivery robots', 'London is a big city in the United Kingdom.']