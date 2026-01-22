import io
import pathlib
import string
import struct
from html import escape
from typing import (
import charset_normalizer  # For str encoding detection
class open_filename(object):
    """
    Context manager that allows opening a filename
    (str or pathlib.PurePath type is supported) and closes it on exit,
    (just like `open`), but does nothing for file-like objects.
    """

    def __init__(self, filename: FileOrName, *args: Any, **kwargs: Any) -> None:
        if isinstance(filename, pathlib.PurePath):
            filename = str(filename)
        if isinstance(filename, str):
            self.file_handler: AnyIO = open(filename, *args, **kwargs)
            self.closing = True
        elif isinstance(filename, io.IOBase):
            self.file_handler = cast(AnyIO, filename)
            self.closing = False
        else:
            raise TypeError('Unsupported input type: %s' % type(filename))

    def __enter__(self) -> AnyIO:
        return self.file_handler

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self.closing:
            self.file_handler.close()