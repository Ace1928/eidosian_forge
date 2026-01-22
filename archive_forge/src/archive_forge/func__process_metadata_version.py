import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import sys
import typing
from typing import (
from . import requirements, specifiers, utils, version as version_module
def _process_metadata_version(self, value: str) -> _MetadataVersion:
    if value not in _VALID_METADATA_VERSIONS:
        raise self._invalid_metadata(f'{value!r} is not a valid metadata version')
    return cast(_MetadataVersion, value)