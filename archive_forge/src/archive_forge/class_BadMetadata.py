import importlib.metadata
from typing import Any, Optional, Protocol, cast
class BadMetadata(ValueError):

    def __init__(self, dist: importlib.metadata.Distribution, *, reason: str) -> None:
        self.dist = dist
        self.reason = reason

    def __str__(self) -> str:
        return f'Bad metadata in {self.dist} ({self.reason})'