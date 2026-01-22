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