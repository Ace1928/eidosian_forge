import operator
import re
from contextlib import contextmanager
from re import Pattern
from ._config import get_run_validators, set_run_validators
from ._make import _AndValidator, and_, attrib, attrs
from .converters import default_if_none
from .exceptions import NotCallableError
@attrs(repr=False, slots=True, hash=True)
class _InValidator:
    options = attrib()

    def __call__(self, inst, attr, value):
        try:
            in_options = value in self.options
        except TypeError:
            in_options = False
        if not in_options:
            msg = f"'{attr.name}' must be in {self.options!r} (got {value!r})"
            raise ValueError(msg, attr, self.options, value)

    def __repr__(self):
        return f'<in_ validator with options {self.options!r}>'