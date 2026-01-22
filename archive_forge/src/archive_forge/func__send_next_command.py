from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
def _send_next_command(self) -> None:
    """
        send out the next command in the queue
        """
    if not self._command_queue:
        self._last_command = None
        return
    command, data = self._command_queue.pop(0)
    self._send_packet(command, data)
    self._last_command = command
    self._last_command_time = time.time()