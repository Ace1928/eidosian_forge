from __future__ import annotations
from abc import abstractmethod
from enum import Enum
from typing import Protocol
class MediaFileStorageError(Exception):
    """Exception class for errors raised by MediaFileStorage.

    When running in "development mode", the full text of these errors
    is displayed in the frontend, so errors should be human-readable
    (and actionable).

    When running in "release mode", errors are redacted on the
    frontend; we instead show a generic "Something went wrong!" message.
    """