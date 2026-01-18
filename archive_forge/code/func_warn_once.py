import functools
import logging
import multiprocessing
import sys
from io import StringIO
from typing import Dict, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
@functools.lru_cache(maxsize=None)
def warn_once() -> None:
    """Warn once about the dangers of PythonREPL."""
    logger.warning('Python REPL can execute arbitrary code. Use with caution.')