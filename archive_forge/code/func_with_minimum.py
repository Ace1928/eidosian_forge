from operator import itemgetter
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Sequence
from . import errors
from .protocol import is_renderable, rich_cast
def with_minimum(self, width: int) -> 'Measurement':
    """Get a RenderableWith where the widths are >= width.

        Args:
            width (int): Minimum desired width.

        Returns:
            Measurement: New Measurement object.
        """
    minimum, maximum = self
    width = max(0, width)
    return Measurement(max(minimum, width), max(maximum, width))