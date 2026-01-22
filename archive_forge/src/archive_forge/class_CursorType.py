from __future__ import annotations
import copy
import warnings
from collections import deque
from typing import (
from bson import RE_TYPE, _convert_raw_document_lists_to_streams
from bson.code import Code
from bson.son import SON
from pymongo import helpers
from pymongo.collation import validate_collation_or_none
from pymongo.common import (
from pymongo.errors import ConnectionFailure, InvalidOperation, OperationFailure
from pymongo.lock import _create_lock
from pymongo.message import (
from pymongo.response import PinnedResponse
from pymongo.typings import _Address, _CollationIn, _DocumentOut, _DocumentType
class CursorType:
    NON_TAILABLE = 0
    'The standard cursor type.'
    TAILABLE = _QUERY_OPTIONS['tailable_cursor']
    'The tailable cursor type.\n\n    Tailable cursors are only for use with capped collections. They are not\n    closed when the last data is retrieved but are kept open and the cursor\n    location marks the final document position. If more data is received\n    iteration of the cursor will continue from the last document received.\n    '
    TAILABLE_AWAIT = TAILABLE | _QUERY_OPTIONS['await_data']
    'A tailable cursor with the await option set.\n\n    Creates a tailable cursor that will wait for a few seconds after returning\n    the full result set so that it can capture and return additional data added\n    during the query.\n    '
    EXHAUST = _QUERY_OPTIONS['exhaust']
    'An exhaust cursor.\n\n    MongoDB will stream batched results to the client without waiting for the\n    client to request each batch, reducing latency.\n    '