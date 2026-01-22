from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def _searchable_form(self) -> 'Language':
    """
        Convert a parsed language tag so that the information it contains is in
        the best form for looking up information in the CLDR.
        """
    if self._searchable is not None:
        return self._searchable
    self._searchable = self._filter_attributes({'language', 'script', 'territory'}).simplify_script().prefer_macrolanguage()
    return self._searchable