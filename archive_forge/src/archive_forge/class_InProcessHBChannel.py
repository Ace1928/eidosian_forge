from typing import List
from jupyter_client.channelsabc import HBChannelABC
class InProcessHBChannel:
    """A dummy heartbeat channel interface for in-process kernels.

    Normally we use the heartbeat to check that the kernel process is alive.
    When the kernel is in-process, that doesn't make sense, but clients still
    expect this interface.
    """
    time_to_dead = 3.0

    def __init__(self, client=None):
        """Initialize the channel."""
        super().__init__()
        self.client = client
        self._is_alive = False
        self._pause = True

    def is_alive(self):
        """Test if the channel is alive."""
        return self._is_alive

    def start(self):
        """Start the channel."""
        self._is_alive = True

    def stop(self):
        """Stop the channel."""
        self._is_alive = False

    def pause(self):
        """Pause the channel."""
        self._pause = True

    def unpause(self):
        """Unpause the channel."""
        self._pause = False

    def is_beating(self):
        """Test if the channel is beating."""
        return not self._pause