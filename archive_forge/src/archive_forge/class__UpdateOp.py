from __future__ import annotations
from typing import (
from bson.raw_bson import RawBSONDocument
from pymongo import helpers
from pymongo.collation import validate_collation_or_none
from pymongo.common import validate_boolean, validate_is_mapping, validate_list
from pymongo.helpers import _gen_index_name, _index_document, _index_list
from pymongo.typings import _CollationIn, _DocumentType, _Pipeline
class _UpdateOp:
    """Private base class for update operations."""
    __slots__ = ('_filter', '_doc', '_upsert', '_collation', '_array_filters', '_hint')

    def __init__(self, filter: Mapping[str, Any], doc: Union[Mapping[str, Any], _Pipeline], upsert: bool, collation: Optional[_CollationIn], array_filters: Optional[list[Mapping[str, Any]]], hint: Optional[_IndexKeyHint]):
        if filter is not None:
            validate_is_mapping('filter', filter)
        if upsert is not None:
            validate_boolean('upsert', upsert)
        if array_filters is not None:
            validate_list('array_filters', array_filters)
        if hint is not None and (not isinstance(hint, str)):
            self._hint: Union[str, SON[str, Any], None] = helpers._index_document(hint)
        else:
            self._hint = hint
        self._filter = filter
        self._doc = doc
        self._upsert = upsert
        self._collation = collation
        self._array_filters = array_filters

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return (other._filter, other._doc, other._upsert, other._collation, other._array_filters, other._hint) == (self._filter, self._doc, self._upsert, self._collation, self._array_filters, self._hint)
        return NotImplemented

    def __repr__(self) -> str:
        return '{}({!r}, {!r}, {!r}, {!r}, {!r}, {!r})'.format(self.__class__.__name__, self._filter, self._doc, self._upsert, self._collation, self._array_filters, self._hint)