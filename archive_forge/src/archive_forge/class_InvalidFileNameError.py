import os
from typing import Optional
class InvalidFileNameError(Exception):
    """InvalidFileNameError class."""
    invalid_str = [os.sep, os.pardir]
    if os.altsep:
        invalid_str.append(os.altsep)

    def __init__(self, filename: str, message: Optional[str]=None):
        self.filename = filename
        self.message = message or f'{filename} cannot contain {self.invalid_str}'
        super().__init__(self.message)