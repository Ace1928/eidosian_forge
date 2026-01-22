from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import (
def _validate_tag_sets(tag_sets: Optional[_TagSets]) -> Optional[_TagSets]:
    """Validate tag sets for a MongoClient."""
    if tag_sets is None:
        return tag_sets
    if not isinstance(tag_sets, (list, tuple)):
        raise TypeError(f'Tag sets {tag_sets!r} invalid, must be a sequence')
    if len(tag_sets) == 0:
        raise ValueError(f'Tag sets {tag_sets!r} invalid, must be None or contain at least one set of tags')
    for tags in tag_sets:
        if not isinstance(tags, abc.Mapping):
            raise TypeError(f'Tag set {tags!r} invalid, must be an instance of dict, bson.son.SON or other type that inherits from collection.Mapping')
    return list(tag_sets)