import json
import re
from collections import (
from typing import (
import attr
from . import (
from .parsing import (
def _build_result_dictionary(self, start: int, end: int, filter_func: Optional[Callable[[HistoryItem], bool]]=None) -> 'OrderedDict[int, HistoryItem]':
    """
        Build history search results
        :param start: start index to search from
        :param end: end index to stop searching (exclusive)
        """
    results: OrderedDict[int, HistoryItem] = OrderedDict()
    for index in range(start, end):
        if filter_func is None or filter_func(self[index]):
            results[index + 1] = self[index]
    return results