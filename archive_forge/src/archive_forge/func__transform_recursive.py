from __future__ import annotations
import io
import base64
import pathlib
from typing import Any, Mapping, TypeVar, cast
from datetime import date, datetime
from typing_extensions import Literal, get_args, override, get_type_hints
import anyio
import pydantic
from ._utils import (
from .._files import is_base64_file_input
from ._typing import (
from .._compat import model_dump, is_typeddict
def _transform_recursive(data: object, *, annotation: type, inner_type: type | None=None) -> object:
    """Transform the given data against the expected type.

    Args:
        annotation: The direct type annotation given to the particular piece of data.
            This may or may not be wrapped in metadata types, e.g. `Required[T]`, `Annotated[T, ...]` etc

        inner_type: If applicable, this is the "inside" type. This is useful in certain cases where the outside type
            is a container type such as `List[T]`. In that case `inner_type` should be set to `T` so that each entry in
            the list can be transformed using the metadata from the container type.

            Defaults to the same value as the `annotation` argument.
    """
    if inner_type is None:
        inner_type = annotation
    stripped_type = strip_annotated_type(inner_type)
    if is_typeddict(stripped_type) and is_mapping(data):
        return _transform_typeddict(data, stripped_type)
    if is_list_type(stripped_type) and is_list(data) or (is_iterable_type(stripped_type) and is_iterable(data) and (not isinstance(data, str))):
        inner_type = extract_type_arg(stripped_type, 0)
        return [_transform_recursive(d, annotation=annotation, inner_type=inner_type) for d in data]
    if is_union_type(stripped_type):
        for subtype in get_args(stripped_type):
            data = _transform_recursive(data, annotation=annotation, inner_type=subtype)
        return data
    if isinstance(data, pydantic.BaseModel):
        return model_dump(data, exclude_unset=True)
    annotated_type = _get_annotated_type(annotation)
    if annotated_type is None:
        return data
    annotations = get_args(annotated_type)[1:]
    for annotation in annotations:
        if isinstance(annotation, PropertyInfo) and annotation.format is not None:
            return _format_data(data, annotation.format, annotation.format_template)
    return data