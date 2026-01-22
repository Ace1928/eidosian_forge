from __future__ import annotations
from base64 import decodebytes, encodebytes
class NotebookReader:
    """A class for reading notebooks."""

    def reads(self, s, **kwargs):
        """Read a notebook from a string."""
        msg = 'loads must be implemented in a subclass'
        raise NotImplementedError(msg)

    def read(self, fp, **kwargs):
        """Read a notebook from a file like object"""
        return self.read(fp.read(), **kwargs)