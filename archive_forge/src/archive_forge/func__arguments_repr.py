from __future__ import annotations
import abc
import datetime
import enum
from collections.abc import MutableMapping as _MutableMapping
from typing import (
from bson.binary import (
from bson.typings import _DocumentType
def _arguments_repr(self) -> str:
    """Representation of the arguments used to create this object."""
    document_class_repr = 'dict' if self.document_class is dict else repr(self.document_class)
    uuid_rep_repr = UUID_REPRESENTATION_NAMES.get(self.uuid_representation, self.uuid_representation)
    return 'document_class={}, tz_aware={!r}, uuid_representation={}, unicode_decode_error_handler={!r}, tzinfo={!r}, type_registry={!r}, datetime_conversion={!s}'.format(document_class_repr, self.tz_aware, uuid_rep_repr, self.unicode_decode_error_handler, self.tzinfo, self.type_registry, self.datetime_conversion)