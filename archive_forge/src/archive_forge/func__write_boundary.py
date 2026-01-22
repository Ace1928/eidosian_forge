import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
def _write_boundary(self):
    """Write the boundary to the end of the buffer."""
    return self._write(self._encoded_boundary)