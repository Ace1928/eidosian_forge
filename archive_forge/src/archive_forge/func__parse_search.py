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
def _parse_search(self, res, **kwargs):
    return Result(res, not kwargs['query']._no_content, duration=kwargs['duration'], has_payload=kwargs['query']._with_payloads, with_scores=kwargs['query']._with_scores)