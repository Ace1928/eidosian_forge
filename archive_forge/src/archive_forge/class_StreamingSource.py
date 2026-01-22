import ctypes
import io
from typing import TYPE_CHECKING, BinaryIO, List, Optional, Union
from pyglet.media.exceptions import MediaException, CannotSeekException
from pyglet.util import next_or_equal_power_of_two
class StreamingSource(Source):
    """A source that is decoded as it is being played.

    The source can only be played once at a time on any
    :class:`~pyglet.media.player.Player`.
    """

    def get_queue_source(self) -> 'StreamingSource':
        """Return the ``Source`` to be used as the source for a player.

        Default implementation returns self.

        Returns:
            :class:`.Source`
        """
        if self.is_player_source:
            raise MediaException('This source is already queued on a player.')
        self.is_player_source = True
        return super().get_queue_source()

    def delete(self) -> None:
        """Release the resources held by this StreamingSource."""
        pass