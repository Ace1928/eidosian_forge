import logging
import os
import queue
import socket
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
import torch.cuda.memory
import torch.cuda.nvtx
import torch.nn as nn
import torch.profiler
import torch.utils.hooks
def _remove_hooks(self) -> None:
    self.hooks_refcount -= 1
    if self.hooks_refcount == 0:
        for h in self.hooks:
            h.remove()