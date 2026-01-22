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
def aliasdel(self, alias: str):
    """
        Removes an alias to a search index

        ### Parameters

        - **alias**: Name of the alias to delete

        For more information see `FT.ALIASDEL <https://redis.io/commands/ft.aliasdel>`_.
        """
    return self.execute_command(ALIAS_DEL_CMD, alias)