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
class _Action(Enum):
    """Specifies the possible actions based on the state of the rendezvous."""
    KEEP_ALIVE = 1
    ADD_TO_PARTICIPANTS = 2
    ADD_TO_WAIT_LIST = 3
    REMOVE_FROM_PARTICIPANTS = 4
    REMOVE_FROM_WAIT_LIST = 5
    MARK_RENDEZVOUS_COMPLETE = 6
    MARK_RENDEZVOUS_CLOSED = 7
    SYNC = 8
    ERROR_CLOSED = 9
    ERROR_TIMEOUT = 10
    FINISH = 11