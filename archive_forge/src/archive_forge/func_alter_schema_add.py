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
def alter_schema_add(self, fields: List[str]):
    """
        Alter the existing search index by adding new fields. The index
        must already exist.

        ### Parameters:

        - **fields**: a list of Field objects to add for the index

        For more information see `FT.ALTER <https://redis.io/commands/ft.alter>`_.
        """
    args = [ALTER_CMD, self.index_name, 'SCHEMA', 'ADD']
    try:
        args += list(itertools.chain(*(f.redis_args() for f in fields)))
    except TypeError:
        args += fields.redis_args()
    return self.execute_command(*args)