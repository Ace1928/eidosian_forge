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
def aliasadd(self, alias: str):
    """
        Alias a search index - will fail if alias already exists

        ### Parameters

        - **alias**: Name of the alias to create

        For more information see `FT.ALIASADD <https://redis.io/commands/ft.aliasadd>`_.
        """
    return self.execute_command(ALIAS_ADD_CMD, alias, self.index_name)