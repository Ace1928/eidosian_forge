import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
def _get_end(self):
    current_pos = self.tell()
    self.seek(0, 2)
    length = self.tell()
    self.seek(current_pos, 0)
    return length