from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
class _TextTypes(enum.Enum):
    """Text types base class that defines base functionality."""

    def __call__(self, *args):
        """Returns a TypedText object using this style."""
        return TypedText(list(args), self)