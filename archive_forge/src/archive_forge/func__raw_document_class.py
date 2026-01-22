from __future__ import annotations
import abc
import datetime
import enum
from collections.abc import MutableMapping as _MutableMapping
from typing import (
from bson.binary import (
from bson.typings import _DocumentType
def _raw_document_class(document_class: Any) -> bool:
    """Determine if a document_class is a RawBSONDocument class."""
    marker = getattr(document_class, '_type_marker', None)
    return marker == _RAW_BSON_DOCUMENT_MARKER