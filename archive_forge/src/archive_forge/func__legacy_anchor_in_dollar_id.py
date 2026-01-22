from __future__ import annotations
from collections.abc import Sequence, Set
from typing import Any, Iterable, Union
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing._attrs import frozen
from referencing._core import (
from referencing.typing import URI, Anchor as AnchorType, Mapping
def _legacy_anchor_in_dollar_id(specification: Specification[Schema], contents: Schema) -> Iterable[Anchor[Schema]]:
    if isinstance(contents, bool):
        return []
    id = contents.get('$id', '')
    if not id.startswith('#'):
        return []
    return [Anchor(name=id[1:], resource=specification.create_resource(contents))]