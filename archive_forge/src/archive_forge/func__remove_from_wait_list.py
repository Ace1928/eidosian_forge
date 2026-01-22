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
def _remove_from_wait_list(self) -> None:
    msg = f"The node '{self._node}' removed itself from the wait list of round {self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
    self._record(message=msg)
    log.debug(msg)
    self._state.wait_list.remove(self._node)
    del self._state.last_heartbeats[self._node]