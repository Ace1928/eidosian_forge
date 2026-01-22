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
def _mk_query_args(self, query, query_params: Union[Dict[str, Union[str, int, float, bytes]], None]):
    args = [self.index_name]
    if isinstance(query, str):
        query = Query(query)
    if not isinstance(query, Query):
        raise ValueError(f'Bad query type {type(query)}')
    args += query.get_args()
    args += self.get_params_args(query_params)
    return (args, query)