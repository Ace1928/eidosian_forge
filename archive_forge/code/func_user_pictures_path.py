from __future__ import annotations
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
@property
def user_pictures_path(self) -> Path:
    """:return: pictures path tied to the user"""
    return Path(self.user_pictures_dir)