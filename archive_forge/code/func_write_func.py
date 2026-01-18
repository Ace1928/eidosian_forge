import re
from io import BytesIO
from .. import errors
def write_func(self, bytes):
    self._write_func(bytes)
    self.current_offset += len(bytes)