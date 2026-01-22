import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
def _calculate_length(self):
    """
        This uses the parts to calculate the length of the body.

        This returns the calculated length so __len__ can be lazy.
        """
    boundary_len = len(self.boundary)
    self._len = sum((boundary_len + total_len(p) + 4 for p in self.parts)) + boundary_len + 4
    return self._len