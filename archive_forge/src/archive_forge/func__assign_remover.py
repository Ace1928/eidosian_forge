from __future__ import annotations
import logging # isort:skip
import sys
import threading
from collections import defaultdict
from traceback import format_exception
from typing import (
import tornado
from tornado import gen
from ..core.types import ID
def _assign_remover(self, callback: Callback, callback_id: ID, removers: Removers, remover: Remover) -> None:
    with self._removers_lock:
        if callback_id in removers:
            raise ValueError('A callback of the same type has already been added with this ID')
        removers[callback_id] = remover