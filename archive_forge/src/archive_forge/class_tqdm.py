import builtins
import copy
import json
import logging
import os
import sys
import threading
import uuid
from typing import Any, Dict, Iterable, Optional
import colorama
import ray
from ray._private.ray_constants import env_bool
from ray.util.debug import log_once
class tqdm:
    """Experimental: Ray distributed tqdm implementation.

    This class lets you use tqdm from any Ray remote task or actor, and have the
    progress centrally reported from the driver. This avoids issues with overlapping
    / conflicting progress bars, as the driver centrally manages tqdm positions.

    Supports a limited subset of tqdm args.
    """

    def __init__(self, iterable: Optional[Iterable]=None, desc: Optional[str]=None, total: Optional[int]=None, position: Optional[int]=None):
        import ray._private.services as services
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except (TypeError, AttributeError):
                total = None
        self._iterable = iterable
        self._desc = desc or ''
        self._total = total
        self._ip = services.get_node_ip_address()
        self._pid = os.getpid()
        self._pos = position or 0
        self._uuid = uuid.uuid4().hex
        self._x = 0
        self._closed = False

    def set_description(self, desc):
        """Implements tqdm.tqdm.set_description."""
        self._desc = desc
        self._dump_state()

    def update(self, n=1):
        """Implements tqdm.tqdm.update."""
        self._x += n
        self._dump_state()

    def close(self):
        """Implements tqdm.tqdm.close."""
        self._closed = True
        if ray is not None:
            self._dump_state()

    def refresh(self):
        """Implements tqdm.tqdm.refresh."""
        self._dump_state()

    @property
    def total(self) -> Optional[int]:
        return self._total

    @total.setter
    def total(self, total: int):
        self._total = total

    def _dump_state(self) -> None:
        if ray._private.worker.global_worker.mode == ray.WORKER_MODE:
            print(json.dumps(self._get_state()) + '\n', end='')
        else:
            instance().process_state_update(copy.deepcopy(self._get_state()))

    def _get_state(self) -> ProgressBarState:
        return {'__magic_token__': RAY_TQDM_MAGIC, 'x': self._x, 'pos': self._pos, 'desc': self._desc, 'total': self._total, 'ip': self._ip, 'pid': self._pid, 'uuid': self._uuid, 'closed': self._closed}

    def __iter__(self):
        if self._iterable is None:
            raise ValueError('No iterable provided')
        for x in iter(self._iterable):
            self.update(1)
            yield x