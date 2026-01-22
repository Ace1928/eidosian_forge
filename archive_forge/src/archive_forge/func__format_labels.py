import math
import sys
from collections import Counter
from pathlib import Path
from typing import (
import numpy
import srsly
import typer
from wasabi import MESSAGES, Printer, msg
from .. import util
from ..compat import Literal
from ..language import Language
from ..morphology import Morphology
from ..pipeline import Morphologizer, SpanCategorizer, TrainablePipe
from ..pipeline._edit_tree_internals.edit_trees import EditTrees
from ..pipeline._parser_internals import nonproj
from ..pipeline._parser_internals.nonproj import DELIMITER
from ..schemas import ConfigSchemaTraining
from ..training import Example, remove_bilu_prefix
from ..training.initialize import get_sourced_components
from ..util import registry, resolve_dot_names
from ..vectors import Mode as VectorsMode
from ._util import (
def _format_labels(labels: Union[Iterable[str], Iterable[Tuple[str, int]]], counts: bool=False) -> str:
    if counts:
        return ', '.join([f"'{l}' ({c})" for l, c in cast(Iterable[Tuple[str, int]], labels)])
    return ', '.join([f"'{l}'" for l in cast(Iterable[str], labels)])