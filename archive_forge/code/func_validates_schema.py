from __future__ import annotations
import functools
from typing import Any, Callable, cast
def validates_schema(fn: Callable[..., Any] | None=None, pass_many: bool=False, pass_original: bool=False, skip_on_field_errors: bool=True) -> Callable[..., Any]:
    """Register a schema-level validator.

    By default it receives a single object at a time, transparently handling the ``many``
    argument passed to the `Schema`'s :func:`~marshmallow.Schema.validate` call.
    If ``pass_many=True``, the raw data (which may be a collection) is passed.

    If ``pass_original=True``, the original data (before unmarshalling) will be passed as
    an additional argument to the method.

    If ``skip_on_field_errors=True``, this validation method will be skipped whenever
    validation errors have been detected when validating fields.

    .. versionchanged:: 3.0.0b1
        ``skip_on_field_errors`` defaults to `True`.

    .. versionchanged:: 3.0.0
        ``partial`` and ``many`` are always passed as keyword arguments to
        the decorated method.
    """
    return set_hook(fn, (VALIDATES_SCHEMA, pass_many), pass_original=pass_original, skip_on_field_errors=skip_on_field_errors)