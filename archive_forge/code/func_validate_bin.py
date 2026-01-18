from __future__ import print_function, unicode_literals
import typing
import six
from ._typing import Text
def validate_bin(self):
    """Validate a mode for opening a binary file.

        Raises:
            ValueError: if the mode contains invalid chars.

        """
    self.validate()
    if 't' in self:
        raise ValueError('mode must be binary')