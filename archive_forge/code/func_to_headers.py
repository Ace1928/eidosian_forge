from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
from uuid import UUID, uuid4
import urllib.parse
from langsmith import schemas as ls_schemas
from langsmith import utils
from langsmith.client import ID_TYPE, RUN_TYPE_T, Client, _dumps_json
def to_headers(self) -> Dict[str, str]:
    """Return the RunTree as a dictionary of headers."""
    headers = {}
    if self.trace_id:
        headers[f'{LANGSMITH_DOTTED_ORDER}'] = self.dotted_order
    baggage = _Baggage(metadata=self.extra.get('metadata', {}), tags=self.tags)
    headers['baggage'] = baggage.to_header()
    return headers