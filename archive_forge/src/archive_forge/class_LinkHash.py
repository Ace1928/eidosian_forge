import functools
import itertools
import logging
import os
import posixpath
import re
import urllib.parse
from dataclasses import dataclass
from typing import (
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.filetypes import WHEEL_EXTENSION
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.misc import (
from pip._internal.utils.models import KeyBasedCompareMixin
from pip._internal.utils.urls import path_to_url, url_to_path
@dataclass(frozen=True)
class LinkHash:
    """Links to content may have embedded hash values. This class parses those.

    `name` must be any member of `_SUPPORTED_HASHES`.

    This class can be converted to and from `ArchiveInfo`. While ArchiveInfo intends to
    be JSON-serializable to conform to PEP 610, this class contains the logic for
    parsing a hash name and value for correctness, and then checking whether that hash
    conforms to a schema with `.is_hash_allowed()`."""
    name: str
    value: str
    _hash_url_fragment_re = re.compile('[#&]({choices})=([^&]*)'.format(choices='|'.join((re.escape(hash_name) for hash_name in _SUPPORTED_HASHES))))

    def __post_init__(self) -> None:
        assert self.name in _SUPPORTED_HASHES

    @classmethod
    @functools.lru_cache(maxsize=None)
    def find_hash_url_fragment(cls, url: str) -> Optional['LinkHash']:
        """Search a string for a checksum algorithm name and encoded output value."""
        match = cls._hash_url_fragment_re.search(url)
        if match is None:
            return None
        name, value = match.groups()
        return cls(name=name, value=value)

    def as_dict(self) -> Dict[str, str]:
        return {self.name: self.value}

    def as_hashes(self) -> Hashes:
        """Return a Hashes instance which checks only for the current hash."""
        return Hashes({self.name: [self.value]})

    def is_hash_allowed(self, hashes: Optional[Hashes]) -> bool:
        """
        Return True if the current hash is allowed by `hashes`.
        """
        if hashes is None:
            return False
        return hashes.is_hash_allowed(self.name, hex_digest=self.value)