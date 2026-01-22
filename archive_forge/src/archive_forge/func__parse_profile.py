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
def _parse_profile(self, res, **kwargs):
    query = kwargs['query']
    if isinstance(query, AggregateRequest):
        result = self._get_aggregate_result(res[0], query, query._cursor)
    else:
        result = Result(res[0], not query._no_content, duration=kwargs['duration'], has_payload=query._with_payloads, with_scores=query._with_scores)
    return (result, parse_to_dict(res[1]))