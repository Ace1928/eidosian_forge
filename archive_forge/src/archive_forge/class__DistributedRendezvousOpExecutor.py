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
class _DistributedRendezvousOpExecutor(_RendezvousOpExecutor):
    """Execute rendezvous operations using a shared state.

    Args:
        node:
            The node descriptor associated with the current rendezvous handler
            instance.
        state_holder:
            The ``RendezvousStateHolder`` to use to sync the rendezvous state
            with other nodes.
        settings:
            The rendezvous settings.
    """
    _node: _NodeDesc
    _state: _RendezvousState
    _state_holder: _RendezvousStateHolder
    _settings: RendezvousSettings

    def __init__(self, node: _NodeDesc, state_holder: _RendezvousStateHolder, settings: RendezvousSettings) -> None:
        self._node = node
        self._state_holder = state_holder
        self._settings = settings

    def _record(self, message: str, node_state: NodeState=NodeState.RUNNING) -> None:
        construct_and_record_rdzv_event(name=f'{self.__class__.__name__}.{get_method_name()}', run_id=self._settings.run_id, message=message, node_state=node_state, hostname=self._node.addr, pid=self._node.pid, local_id=self._node.local_id)

    def run(self, state_handler: Callable[[_RendezvousContext, float], _Action], deadline: float) -> None:
        """See base class."""
        action = None
        while action != _Action.FINISH:
            has_set = self._state_holder.sync()
            if has_set is not None:
                if has_set:
                    msg = f"The node '{self._node}' has successfully synced its local changes with other nodes in the rendezvous '{self._settings.run_id}'."
                else:
                    msg = f"The node '{self._node}' has a stale state and failed to sync its local changes with other nodes in the rendezvous '{self._settings.run_id}'."
                self._record(message=msg)
                log.debug(msg)
            self._state = self._state_holder.state
            ctx = _RendezvousContext(self._node, self._state, self._settings)
            action = state_handler(ctx, deadline)
            if action == _Action.FINISH:
                continue
            if action == _Action.ERROR_CLOSED:
                raise RendezvousClosedError()
            if action == _Action.ERROR_TIMEOUT:
                raise RendezvousTimeoutError()
            if action == _Action.SYNC:
                _delay(seconds=1)
            else:
                if action == _Action.KEEP_ALIVE:
                    self._keep_alive()
                elif action == _Action.ADD_TO_PARTICIPANTS:
                    self._add_to_participants()
                elif action == _Action.ADD_TO_WAIT_LIST:
                    self._add_to_wait_list()
                elif action == _Action.REMOVE_FROM_PARTICIPANTS:
                    self._remove_from_participants()
                elif action == _Action.REMOVE_FROM_WAIT_LIST:
                    self._remove_from_wait_list()
                elif action == _Action.MARK_RENDEZVOUS_COMPLETE:
                    self._mark_rendezvous_complete()
                elif action == _Action.MARK_RENDEZVOUS_CLOSED:
                    self._mark_rendezvous_closed()
                self._state_holder.mark_dirty()

    def _keep_alive(self) -> None:
        msg = f"The node '{self._node}' updated its keep-alive heartbeat time for the rendezvous '{self._settings.run_id}'. Pending sync."
        self._record(message=msg)
        log.debug(msg)
        self._state.last_heartbeats[self._node] = datetime.utcnow()

    def _add_to_participants(self) -> None:
        msg = f"The node '{self._node}' added itself to the participants of round {self._state.round} of the rendezvous '{self._settings.run_id}'. Pending sync."
        self._record(message=msg)
        log.debug(msg)
        state = self._state
        try:
            state.wait_list.remove(self._node)
        except KeyError:
            pass
        state.participants[self._node] = 0
        self._keep_alive()
        if len(state.participants) == self._settings.min_nodes:
            state.deadline = datetime.utcnow() + self._settings.timeout.last_call
        if len(state.participants) == self._settings.max_nodes:
            self._mark_rendezvous_complete()

    def _add_to_wait_list(self) -> None:
        msg = f"The node '{self._node}' added itself to the wait list of round {self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        self._record(message=msg)
        log.debug(msg)
        self._state.wait_list.add(self._node)
        self._keep_alive()

    def _remove_from_participants(self) -> None:
        msg = f"The node '{self._node}' removed itself from the participants of round {self._state.round} of the rendezvous '{self._settings.run_id}'. Pending sync."
        self._record(message=msg)
        log.debug(msg)
        state = self._state
        del state.participants[self._node]
        del state.last_heartbeats[self._node]
        _remove_participant_epilogue(state, self._settings)

    def _remove_from_wait_list(self) -> None:
        msg = f"The node '{self._node}' removed itself from the wait list of round {self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        self._record(message=msg)
        log.debug(msg)
        self._state.wait_list.remove(self._node)
        del self._state.last_heartbeats[self._node]

    def _mark_rendezvous_complete(self) -> None:
        msg = f"The node '{self._node}' marked round {self._state.round} of the rendezvous '{self._settings.run_id}' as complete. Pending sync."
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        log.debug(msg)
        state = self._state
        state.complete = True
        state.deadline = None
        for rank, node in enumerate(sorted(state.participants)):
            state.participants[node] = rank

    def _mark_rendezvous_closed(self) -> None:
        msg = f"The node '{self._node}' marked the rendezvous '{self._settings.run_id}' as closed. Pending sync."
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        log.debug(msg)
        self._state.closed = True