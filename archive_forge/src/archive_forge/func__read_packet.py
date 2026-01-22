from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
def _read_packet(self) -> tuple[int, bytearray] | None:
    """
        low-level packet reading.
        returns (command/report code, data) or None

        This method stored data read and tries to resync when bad data
        is received.
        """
    self._unprocessed += self._device.read()
    while True:
        try:
            command, data, unprocessed = self._parse_data(self._unprocessed)
            self._unprocessed = unprocessed
        except self.MoreDataRequired:
            return None
        except self.InvalidPacket:
            self._unprocessed = self._unprocessed[1:]
        else:
            return (command, data)