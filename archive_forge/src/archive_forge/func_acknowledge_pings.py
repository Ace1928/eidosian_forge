from __future__ import annotations
import contextlib
import logging
import random
import socket
import struct
import threading
import uuid
from types import TracebackType
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Type, Union
from ..exceptions import ConnectionClosed, ConnectionClosedOK, ProtocolError
from ..frames import DATA_OPCODES, BytesLike, CloseCode, Frame, Opcode, prepare_ctrl
from ..http11 import Request, Response
from ..protocol import CLOSED, OPEN, Event, Protocol, State
from ..typing import Data, LoggerLike, Subprotocol
from .messages import Assembler
from .utils import Deadline
def acknowledge_pings(self, data: bytes) -> None:
    """
        Acknowledge pings when receiving a pong.

        """
    with self.protocol_mutex:
        if data not in self.pings:
            return
        ping_id = None
        ping_ids = []
        for ping_id, ping in self.pings.items():
            ping_ids.append(ping_id)
            ping.set()
            if ping_id == data:
                break
        else:
            raise AssertionError('solicited pong not found in pings')
        for ping_id in ping_ids:
            del self.pings[ping_id]