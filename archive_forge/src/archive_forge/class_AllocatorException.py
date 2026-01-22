from typing import TYPE_CHECKING, Tuple, Optional
import pyglet
class AllocatorException(Exception):
    """The allocator does not have sufficient free space for the requested
    image size."""
    pass