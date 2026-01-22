import inspect
import logging
import os
import pickle
import socket
import threading
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast
from torch.distributed import PrefixStore, Store
from torch.distributed.elastic.events import (
from .api import (
from .utils import _delay, _PeriodicTimer
class _BackendRendezvousStateHolder(_RendezvousStateHolder):
    """Hold the rendezvous state synced with other nodes via a backend.

    Args:
        backend:
            The rendezvous backend to use.
        settings:
            The rendezvous settings.
        cache_duration:
            The amount of time, in seconds, to cache the last rendezvous state
            before requesting it from the backend again.
    """
    _backend: RendezvousBackend
    _state: _RendezvousState
    _settings: RendezvousSettings
    _cache_duration: int
    _token: Token
    _dirty: bool
    _last_sync_time: float
    _dead_nodes: List[_NodeDesc]

    def __init__(self, backend: RendezvousBackend, settings: RendezvousSettings, cache_duration: int=1) -> None:
        self._backend = backend
        self._state = _RendezvousState()
        self._settings = settings
        self._cache_duration = cache_duration
        self._token = None
        self._dirty = False
        self._last_sync_time = -1
        self._dead_nodes = []

    def _record(self, message: str, node_state: NodeState=NodeState.RUNNING):
        construct_and_record_rdzv_event(name=f'{self.__class__.__name__}.{get_method_name()}', run_id=self._settings.run_id, message=message, node_state=node_state)

    @property
    def state(self) -> _RendezvousState:
        """See base class."""
        return self._state

    def sync(self) -> Optional[bool]:
        """See base class."""
        state_bits: Optional[bytes] = None
        token = None
        has_set: Optional[bool]
        if self._dirty:
            has_set = False
            state_bits = pickle.dumps(self._state)
            set_response = self._backend.set_state(state_bits, self._token)
            if set_response is not None:
                state_bits, token, has_set = set_response
        else:
            has_set = None
            if self._cache_duration > 0:
                if self._last_sync_time >= max(time.monotonic() - self._cache_duration, 0):
                    return None
            get_response = self._backend.get_state()
            if get_response is not None:
                state_bits, token = get_response
        if state_bits is not None:
            try:
                self._state = pickle.loads(state_bits)
            except pickle.PickleError as exc:
                raise RendezvousStateError('The rendezvous state is corrupt. See inner exception for details.') from exc
        else:
            self._state = _RendezvousState()
        if has_set and self._dead_nodes and log.isEnabledFor(logging.DEBUG):
            node_list = ', '.join((f"'{dead_node}'" for dead_node in self._dead_nodes))
            msg = f"As part of the sync operation the node(s) {node_list} have been removed from the rendezvous '{self._settings.run_id}' since they had no heartbeat."
            self._record(message=msg)
            log.debug(msg)
        self._token = token
        self._dirty = False
        self._last_sync_time = time.monotonic()
        self._sanitize()
        return has_set

    def _sanitize(self) -> None:
        state = self._state
        expire_time = datetime.utcnow() - self._settings.keep_alive_interval * self._settings.keep_alive_max_attempt
        self._dead_nodes = [node for node, last_heartbeat in state.last_heartbeats.items() if last_heartbeat < expire_time]
        participant_removed = False
        for dead_node in self._dead_nodes:
            del state.last_heartbeats[dead_node]
            try:
                del state.participants[dead_node]
                participant_removed = True
            except KeyError:
                pass
            try:
                state.wait_list.remove(dead_node)
            except KeyError:
                pass
        if participant_removed:
            _remove_participant_epilogue(state, self._settings)

    def mark_dirty(self) -> None:
        """See base class.

        If the local rendezvous state is dirty, the next sync call will try to
        write the changes back to the backend. However this attempt might fail
        if another node, which had the same state, also made changes and wrote
        them before us.
        """
        self._dirty = True