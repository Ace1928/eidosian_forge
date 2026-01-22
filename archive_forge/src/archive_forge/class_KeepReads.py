from __future__ import annotations
import socket
from twisted.internet import udp
from twisted.internet.protocol import DatagramProtocol
from twisted.python.runtime import platformType
from twisted.trial import unittest
class KeepReads(DatagramProtocol):
    """
    Accumulate reads in a list.
    """

    def __init__(self) -> None:
        self.reads: list[bytes] = []

    def datagramReceived(self, data: bytes, addr: object) -> None:
        self.reads.append(data)