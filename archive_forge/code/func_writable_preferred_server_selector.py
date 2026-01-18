from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, TypeVar, cast
from pymongo.server_type import SERVER_TYPE
def writable_preferred_server_selector(selection: Selection) -> Selection:
    """Like PrimaryPreferred but doesn't use tags or latency."""
    return writable_server_selector(selection) or secondary_server_selector(selection)