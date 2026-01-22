import abc
import itertools
import re
from typing import Callable, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union
from .utils import canonicalize_version
from .version import Version
def _compare_compatible(self, prospective: Version, spec: str) -> bool:
    prefix = _version_join(list(itertools.takewhile(_is_not_suffix, _version_split(spec)))[:-1])
    prefix += '.*'
    return self._get_operator('>=')(prospective, spec) and self._get_operator('==')(prospective, prefix)