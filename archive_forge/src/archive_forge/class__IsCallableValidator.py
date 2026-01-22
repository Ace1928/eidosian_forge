import operator
import re
from contextlib import contextmanager
from re import Pattern
from ._config import get_run_validators, set_run_validators
from ._make import _AndValidator, and_, attrib, attrs
from .converters import default_if_none
from .exceptions import NotCallableError
@attrs(repr=False, slots=False, hash=True)
class _IsCallableValidator:

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not callable(value):
            message = "'{name}' must be callable (got {value!r} that is a {actual!r})."
            raise NotCallableError(msg=message.format(name=attr.name, value=value, actual=value.__class__), value=value)

    def __repr__(self):
        return '<is_callable validator>'