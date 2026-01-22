from __future__ import annotations as _annotations
import dataclasses
import inspect
import typing
from copy import copy
from dataclasses import Field as DataclassField
from functools import cached_property
from typing import Any, ClassVar
from warnings import warn
import annotated_types
import typing_extensions
from pydantic_core import PydanticUndefined
from typing_extensions import Literal, Unpack
from . import types
from ._internal import _decorators, _fields, _generics, _internal_dataclass, _repr, _typing_extra, _utils
from .aliases import AliasChoices, AliasPath
from .config import JsonDict
from .errors import PydanticUserError
from .warnings import PydanticDeprecatedSince20
@staticmethod
def _collect_metadata(kwargs: dict[str, Any]) -> list[Any]:
    """Collect annotations from kwargs.

        Args:
            kwargs: Keyword arguments passed to the function.

        Returns:
            A list of metadata objects - a combination of `annotated_types.BaseMetadata` and
                `PydanticMetadata`.
        """
    metadata: list[Any] = []
    general_metadata = {}
    for key, value in list(kwargs.items()):
        try:
            marker = FieldInfo.metadata_lookup[key]
        except KeyError:
            continue
        del kwargs[key]
        if value is not None:
            if marker is None:
                general_metadata[key] = value
            else:
                metadata.append(marker(value))
    if general_metadata:
        metadata.append(_fields.pydantic_general_metadata(**general_metadata))
    return metadata