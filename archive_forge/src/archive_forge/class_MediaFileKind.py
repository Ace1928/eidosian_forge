from __future__ import annotations
from abc import abstractmethod
from enum import Enum
from typing import Protocol
class MediaFileKind(Enum):
    MEDIA = 'media'
    DOWNLOADABLE = 'downloadable'