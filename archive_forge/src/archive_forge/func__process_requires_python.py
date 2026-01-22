import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import sys
import typing
from typing import (
from . import requirements, specifiers, utils, version as version_module
def _process_requires_python(self, value: str) -> specifiers.SpecifierSet:
    try:
        return specifiers.SpecifierSet(value)
    except specifiers.InvalidSpecifier as exc:
        raise self._invalid_metadata(f'{value!r} is invalid for {{field}}', cause=exc)