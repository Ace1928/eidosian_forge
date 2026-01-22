import abc
import functools
import itertools
import re
import warnings
from typing import (
from .utils import canonicalize_version
from .version import LegacyVersion, Version, parse
class LegacySpecifier(_IndividualSpecifier):
    _regex_str = '\n        (?P<operator>(==|!=|<=|>=|<|>))\n        \\s*\n        (?P<version>\n            [^,;\\s)]* # Since this is a "legacy" specifier, and the version\n                      # string can be just about anything, we match everything\n                      # except for whitespace, a semi-colon for marker support,\n                      # a closing paren since versions can be enclosed in\n                      # them, and a comma since it\'s a version separator.\n        )\n        '
    _regex = re.compile('^\\s*' + _regex_str + '\\s*$', re.VERBOSE | re.IGNORECASE)
    _operators = {'==': 'equal', '!=': 'not_equal', '<=': 'less_than_equal', '>=': 'greater_than_equal', '<': 'less_than', '>': 'greater_than'}

    def __init__(self, spec: str='', prereleases: Optional[bool]=None) -> None:
        super().__init__(spec, prereleases)
        warnings.warn('Creating a LegacyVersion has been deprecated and will be removed in the next major release', DeprecationWarning)

    def _coerce_version(self, version: UnparsedVersion) -> LegacyVersion:
        if not isinstance(version, LegacyVersion):
            version = LegacyVersion(str(version))
        return version

    def _compare_equal(self, prospective: LegacyVersion, spec: str) -> bool:
        return prospective == self._coerce_version(spec)

    def _compare_not_equal(self, prospective: LegacyVersion, spec: str) -> bool:
        return prospective != self._coerce_version(spec)

    def _compare_less_than_equal(self, prospective: LegacyVersion, spec: str) -> bool:
        return prospective <= self._coerce_version(spec)

    def _compare_greater_than_equal(self, prospective: LegacyVersion, spec: str) -> bool:
        return prospective >= self._coerce_version(spec)

    def _compare_less_than(self, prospective: LegacyVersion, spec: str) -> bool:
        return prospective < self._coerce_version(spec)

    def _compare_greater_than(self, prospective: LegacyVersion, spec: str) -> bool:
        return prospective > self._coerce_version(spec)