import itertools
import operator
import warnings
from abc import ABCMeta, abstractmethod
from collections import deque
from collections.abc import MutableSequence
from copy import deepcopy
from functools import partial as _partial
from functools import reduce
from operator import itemgetter
from types import GeneratorType
from kombu.utils.functional import fxrange, reprcall
from kombu.utils.objects import cached_property
from kombu.utils.uuid import uuid
from vine import barrier
from celery._state import current_app
from celery.exceptions import CPendingDeprecationWarning
from celery.result import GroupResult, allow_join_result
from celery.utils import abstract
from celery.utils.collections import ChainMap
from celery.utils.functional import _regen
from celery.utils.functional import chunks as _chunks
from celery.utils.functional import is_list, maybe_list, regen, seq_concat_item, seq_concat_seq
from celery.utils.objects import getitem_property
from celery.utils.text import remove_repeating_from_task, truncate
def _stamp_headers(self, visitor_headers=None, append_stamps=False, self_headers=True, **headers):
    """Collect all stamps from visitor, headers and self,
        and return an idempotent dictionary of stamps.

        .. versionadded:: 5.3

        Arguments:
            visitor_headers (Dict): Stamps from a visitor method.
            append_stamps (bool):
                If True, duplicated stamps will be appended to a list.
                If False, duplicated stamps will be replaced by the last stamp.
            self_headers (bool):
                If True, stamps from self.options will be added.
                If False, stamps from self.options will be ignored.
            headers (Dict): Stamps that should be added to headers.

        Returns:
            Dict: Merged stamps.
        """
    headers = headers.copy()
    if 'stamped_headers' not in headers:
        headers['stamped_headers'] = list(headers.keys())
    if visitor_headers is not None:
        visitor_headers = visitor_headers or {}
        if 'stamped_headers' not in visitor_headers:
            visitor_headers['stamped_headers'] = list(visitor_headers.keys())
        _merge_dictionaries(headers, visitor_headers, aggregate_duplicates=append_stamps)
        headers['stamped_headers'] = list(set(headers['stamped_headers']))
    if self_headers:
        stamped_headers = set(headers.get('stamped_headers', []))
        stamped_headers.update(self.options.get('stamped_headers', []))
        headers['stamped_headers'] = list(stamped_headers)
        redacted_options = {k: v for k, v in self.options.items() if k in headers['stamped_headers']}
        _merge_dictionaries(headers, redacted_options, aggregate_duplicates=append_stamps)
        headers['stamped_headers'] = list(set(headers['stamped_headers']))
    return headers