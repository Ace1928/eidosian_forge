from __future__ import annotations
import gc
import atexit
import asyncio
import contextlib
import collections.abc
from lazyops.utils.lazy import lazy_import, get_keydb_enabled
from lazyops.utils.logs import logger, null_logger
from lazyops.utils.pooler import ThreadPooler
from typing import Any, Dict, Optional, Union, Iterable, List, Type, Set, Callable, Mapping, MutableMapping, Tuple, TypeVar, overload, TYPE_CHECKING
from .backends import LocalStatefulBackend, RedisStatefulBackend, StatefulBackendT
from .serializers import ObjectValue
from .addons import (
from .debug import get_autologger
def _save_mutation_objects(self, *keys: str):
    """
        Saves the Mutation Objects
        """
    if not self._mutation_tracker:
        return
    if keys:
        for key in keys:
            if key in self._mutation_tracker:
                self.base.set(key, self._mutation_tracker[key])
                self._clear_from_mutation_tracker(key)
    else:
        autologger.info(f'_save_mutation_objects: {list(self._mutation_tracker.keys())}')
        self.base.set_batch(self._mutation_tracker)
        self._mutation_tracker = {}
        self._mutation_hashes = {}
    gc.collect()