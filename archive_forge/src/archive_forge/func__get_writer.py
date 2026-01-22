import sys
import re
import warnings
import io
import collections
import collections.abc
import contextlib
import weakref
from . import ElementPath
fromstring = XML
@contextlib.contextmanager
def _get_writer(file_or_filename, encoding):
    try:
        write = file_or_filename.write
    except AttributeError:
        if encoding.lower() == 'unicode':
            encoding = 'utf-8'
        with open(file_or_filename, 'w', encoding=encoding, errors='xmlcharrefreplace') as file:
            yield (file.write, encoding)
    else:
        if encoding.lower() == 'unicode':
            yield (write, getattr(file_or_filename, 'encoding', None) or 'utf-8')
        else:
            with contextlib.ExitStack() as stack:
                if isinstance(file_or_filename, io.BufferedIOBase):
                    file = file_or_filename
                elif isinstance(file_or_filename, io.RawIOBase):
                    file = io.BufferedWriter(file_or_filename)
                    stack.callback(file.detach)
                else:
                    file = io.BufferedIOBase()
                    file.writable = lambda: True
                    file.write = write
                    try:
                        file.seekable = file_or_filename.seekable
                        file.tell = file_or_filename.tell
                    except AttributeError:
                        pass
                file = io.TextIOWrapper(file, encoding=encoding, errors='xmlcharrefreplace', newline='\n')
                stack.callback(file.detach)
                yield (file.write, encoding)