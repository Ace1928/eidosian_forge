from operator import itemgetter
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Sequence
from . import errors
from .protocol import is_renderable, rich_cast
def with_maximum(self, width: int) -> 'Measurement':
    """Get a RenderableWith where the widths are <= width.

        Args:
            width (int): Maximum desired width.

        Returns:
            Measurement: New Measurement object.
        """
    minimum, maximum = self
    return Measurement(min(minimum, width), min(maximum, width))