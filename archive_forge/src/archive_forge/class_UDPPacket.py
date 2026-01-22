from __future__ import annotations
import contextlib
import errno
import ipaddress
import os
import socket
import sys
from typing import (
import attrs
import trio
from trio._util import NoPublicConstructor, final
@attrs.frozen
class UDPPacket:
    source: UDPEndpoint
    destination: UDPEndpoint
    payload: bytes = attrs.field(repr=lambda p: p.hex())

    def reply(self, payload: bytes) -> UDPPacket:
        return UDPPacket(source=self.destination, destination=self.source, payload=payload)