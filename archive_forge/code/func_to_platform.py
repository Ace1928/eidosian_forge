from __future__ import print_function, unicode_literals
import typing
import six
from ._typing import Text
def to_platform(self):
    """Get a mode string for the current platform.

        Currently, this just removes the 'x' on PY2 because PY2 doesn't
        support exclusive mode.

        """
    return self._mode.replace('x', 'w') if six.PY2 else self._mode