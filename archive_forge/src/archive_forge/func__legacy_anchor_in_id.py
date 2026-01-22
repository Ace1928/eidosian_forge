from __future__ import annotations
from collections.abc import Sequence, Set
from typing import Any, Iterable, Union
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing._attrs import frozen
from referencing._core import (
from referencing.typing import URI, Anchor as AnchorType, Mapping
def _legacy_anchor_in_id(specification: Specification[ObjectSchema], contents: ObjectSchema) -> Iterable[Anchor[ObjectSchema]]:
    id = contents.get('id', '')
    if not id.startswith('#'):
        return []
    return [Anchor(name=id[1:], resource=specification.create_resource(contents))]