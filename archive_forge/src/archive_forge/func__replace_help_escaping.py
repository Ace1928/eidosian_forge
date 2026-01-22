import io as StringIO
import re
from typing import Dict, Iterable, List, Match, Optional, TextIO, Tuple
from .metrics_core import Metric
from .samples import Sample
def _replace_help_escaping(s: str) -> str:
    return HELP_ESCAPING_RE.sub(replace_escape_sequence, s)