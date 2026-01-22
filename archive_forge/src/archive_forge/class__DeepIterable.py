import operator
import re
from contextlib import contextmanager
from re import Pattern
from ._config import get_run_validators, set_run_validators
from ._make import _AndValidator, and_, attrib, attrs
from .converters import default_if_none
from .exceptions import NotCallableError
@attrs(repr=False, slots=True, hash=True)
class _DeepIterable:
    member_validator = attrib(validator=is_callable())
    iterable_validator = attrib(default=None, validator=optional(is_callable()))

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if self.iterable_validator is not None:
            self.iterable_validator(inst, attr, value)
        for member in value:
            self.member_validator(inst, attr, member)

    def __repr__(self):
        iterable_identifier = '' if self.iterable_validator is None else f' {self.iterable_validator!r}'
        return f'<deep_iterable validator for{iterable_identifier} iterables of {self.member_validator!r}>'