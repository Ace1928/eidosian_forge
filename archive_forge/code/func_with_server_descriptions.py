from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, TypeVar, cast
from pymongo.server_type import SERVER_TYPE
def with_server_descriptions(self, server_descriptions: list[ServerDescription]) -> Selection:
    return Selection(self.topology_description, server_descriptions, self.common_wire_version, self.primary)