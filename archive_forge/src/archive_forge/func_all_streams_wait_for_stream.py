import enum
import functools
import inspect
import io
import logging
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TypeVar
import torch
import torch.utils._cuda_trace as cuda_trace
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
def all_streams_wait_for_stream(self, stream: StreamId) -> None:
    self._ensure_stream_exists(stream)
    for state in self.current_sync_states.values():
        self._state_wait_for_other(state, self.current_sync_states[stream])
    self._state_wait_for_other(self.host_sync_state, self.current_sync_states[stream])