import os
from typing import Optional
class ParentPathError(Exception):
    """ParentPathError class."""

    def __init__(self, path: str, parent_path: str, message: Optional[str]=None):
        self.path = path
        self.sandbox_path = parent_path
        self.message = message or f'{path} is not a part of {parent_path}'
        super().__init__(self.message)