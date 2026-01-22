import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
def _prepare_parts(self):
    """This uses the fields provided by the user and creates Part objects.

        It populates the `parts` attribute and uses that to create a
        generator for iteration.
        """
    enc = self.encoding
    self.parts = [Part.from_field(f, enc) for f in self._iter_fields()]
    self._iter_parts = iter(self.parts)