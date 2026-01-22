from __future__ import division
from .exceptions import FlowControlError
def _maybe_update_window(self):
    """
        Run the algorithm.

        Our current algorithm can be described like this.

        1. If no bytes have been processed, we immediately return 0. There is
           no meaningful way for us to hand space in the window back to the
           remote peer, so let's not even try.
        2. If there is no space in the flow control window, and we have
           processed at least 1024 bytes (or 1/4 of the window, if the window
           is smaller), we will emit a window update frame. This is to avoid
           the risk of blocking a stream altogether.
        3. If there is space in the flow control window, and we have processed
           at least 1/2 of the window worth of bytes, we will emit a window
           update frame. This is to minimise the number of window update frames
           we have to emit.

        In a healthy system with large flow control windows, this will
        irregularly emit WINDOW_UPDATE frames. This prevents us starving the
        connection by emitting eleventy bajillion WINDOW_UPDATE frames,
        especially in situations where the remote peer is sending a lot of very
        small DATA frames.
        """
    if not self._bytes_processed:
        return None
    max_increment = self.max_window_size - self.current_window_size
    increment = 0
    if self.current_window_size == 0 and self._bytes_processed > min(1024, self.max_window_size // 4):
        increment = min(self._bytes_processed, max_increment)
        self._bytes_processed = 0
    elif self._bytes_processed >= self.max_window_size // 2:
        increment = min(self._bytes_processed, max_increment)
        self._bytes_processed = 0
    self.current_window_size += increment
    return increment