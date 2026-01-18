import inspect
import pathlib
import sys
import typing
from collections import defaultdict
from types import CodeType
from typing import Dict, Iterable, List, Optional
import torch
def trace_logger(self) -> JitTypeTraceStoreLogger:
    """Return a JitCallTraceStoreLogger that logs to the configured trace store."""
    return JitTypeTraceStoreLogger(self.trace_store())