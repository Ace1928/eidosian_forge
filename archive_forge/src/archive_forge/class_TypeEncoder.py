from __future__ import annotations
import abc
import datetime
import enum
from collections.abc import MutableMapping as _MutableMapping
from typing import (
from bson.binary import (
from bson.typings import _DocumentType
class TypeEncoder(abc.ABC):
    """Base class for defining type codec classes which describe how a
    custom type can be transformed to one of the types BSON understands.

    Codec classes must implement the ``python_type`` attribute, and the
    ``transform_python`` method to support encoding.

    See :ref:`custom-type-type-codec` documentation for an example.
    """

    @abc.abstractproperty
    def python_type(self) -> Any:
        """The Python type to be converted into something serializable."""

    @abc.abstractmethod
    def transform_python(self, value: Any) -> Any:
        """Convert the given Python object into something serializable."""