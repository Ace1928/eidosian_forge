from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import CreatePrimaryIPResponse, PrimaryIP
class PrimaryIPsPageResult(NamedTuple):
    primary_ips: list[BoundPrimaryIP]
    meta: Meta | None