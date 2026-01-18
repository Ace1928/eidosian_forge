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
def update_profilers_on_step(self) -> None:
    for p in self.profilers:
        if p.iter_begin <= self.done_steps and self.done_steps < p.iter_end:
            if p.object is None:
                o = p.cls(self)
                logging.info(f'Starting {p.cls.__name__} profiler...')
                o.__enter__()
                p.object = o
            else:
                p.object.step()
        elif p.object is not None:
            o = p.object
            p.object = None
            logging.info(f'Shutting down {p.cls.__name__} profiler...')
            o.__exit__(None, None, None)