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
def _get_examples_without_label(data: Sequence[Example], label: str, component: Literal['ner', 'spancat']='ner', spans_key: Optional[str]='sc') -> int:
    count = 0
    for eg in data:
        if component == 'ner':
            labels = [remove_bilu_prefix(label) for label in eg.get_aligned_ner() if label not in ('O', '-', None)]
        if component == 'spancat':
            labels = [span.label_ for span in eg.reference.spans[spans_key]] if spans_key in eg.reference.spans else []
        if label not in labels:
            count += 1
    return count