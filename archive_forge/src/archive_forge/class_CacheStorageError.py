from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, Protocol
class CacheStorageError(Exception):
    """Base exception raised by the cache storage"""