from abc import ABC, abstractmethod
from .header import Field
Saves streamlines to a filename or file-like object.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            opened and ready to write.
        