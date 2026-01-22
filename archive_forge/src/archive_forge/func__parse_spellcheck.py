import itertools
import time
from typing import Dict, List, Optional, Union
from redis.client import Pipeline
from redis.utils import deprecated_function
from ..helpers import get_protocol_version, parse_to_dict
from ._util import to_string
from .aggregation import AggregateRequest, AggregateResult, Cursor
from .document import Document
from .query import Query
from .result import Result
from .suggestion import SuggestionParser
def _parse_spellcheck(self, res, **kwargs):
    corrections = {}
    if res == 0:
        return corrections
    for _correction in res:
        if isinstance(_correction, int) and _correction == 0:
            continue
        if len(_correction) != 3:
            continue
        if not _correction[2]:
            continue
        if not _correction[2][0]:
            continue
        corrections[_correction[1]] = [{'score': _item[0], 'suggestion': _item[1]} for _item in _correction[2]]
    return corrections