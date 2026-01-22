import abc
import itertools
import re
from typing import Callable, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union
from .utils import canonicalize_version
from .version import Version
@property
def _canonical_spec(self) -> Tuple[str, str]:
    canonical_version = canonicalize_version(self._spec[1], strip_trailing_zero=self._spec[0] != '~=')
    return (self._spec[0], canonical_version)