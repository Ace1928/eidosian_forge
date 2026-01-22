from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence
from langchain_core.outputs import Generation
from langchain_core.runnables import run_in_executor
Clear cache that can take additional keyword arguments.