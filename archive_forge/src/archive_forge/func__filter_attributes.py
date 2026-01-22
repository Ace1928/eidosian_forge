from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def _filter_attributes(self, keyset: Iterable[str]) -> 'Language':
    """
        Return a copy of this object with a subset of its attributes set.
        """
    filtered = self._filter_keys(self.to_dict(), keyset)
    return Language.make(**filtered)