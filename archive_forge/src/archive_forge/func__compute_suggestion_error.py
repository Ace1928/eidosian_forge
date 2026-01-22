from __future__ import annotations
import collections.abc
import sys
import textwrap
import traceback
from functools import singledispatch
from types import TracebackType
from typing import Any, List, Optional
from ._exceptions import BaseExceptionGroup
def _compute_suggestion_error(exc_value, tb):
    wrong_name = getattr(exc_value, 'name', None)
    if wrong_name is None or not isinstance(wrong_name, str):
        return None
    if isinstance(exc_value, AttributeError):
        obj = getattr(exc_value, 'obj', _SENTINEL)
        if obj is _SENTINEL:
            return None
        obj = exc_value.obj
        try:
            d = dir(obj)
        except Exception:
            return None
    else:
        assert isinstance(exc_value, NameError)
        if tb is None:
            return None
        while tb.tb_next is not None:
            tb = tb.tb_next
        frame = tb.tb_frame
        d = list(frame.f_locals) + list(frame.f_globals) + list(frame.f_builtins)
    if len(d) > _MAX_CANDIDATE_ITEMS:
        return None
    wrong_name_len = len(wrong_name)
    if wrong_name_len > _MAX_STRING_SIZE:
        return None
    best_distance = wrong_name_len
    suggestion = None
    for possible_name in d:
        if possible_name == wrong_name:
            continue
        max_distance = (len(possible_name) + wrong_name_len + 3) * _MOVE_COST // 6
        max_distance = min(max_distance, best_distance - 1)
        current_distance = _levenshtein_distance(wrong_name, possible_name, max_distance)
        if current_distance > max_distance:
            continue
        if not suggestion or current_distance < best_distance:
            suggestion = possible_name
            best_distance = current_distance
    return suggestion