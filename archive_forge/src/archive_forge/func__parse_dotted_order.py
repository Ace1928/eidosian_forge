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
def _parse_dotted_order(dotted_order: str) -> List[Tuple[datetime, UUID]]:
    """Parse the dotted order string."""
    parts = dotted_order.split('.')
    return [(datetime.strptime(part[:-36], '%Y%m%dT%H%M%S%fZ'), UUID(part[-36:])) for part in parts]