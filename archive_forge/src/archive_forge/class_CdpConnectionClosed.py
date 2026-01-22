import contextvars
import importlib
import itertools
import json
import logging
import pathlib
import typing
from collections import defaultdict
from contextlib import asynccontextmanager
from contextlib import contextmanager
from dataclasses import dataclass
import trio
from trio_websocket import ConnectionClosed as WsConnectionClosed
from trio_websocket import connect_websocket_url
class CdpConnectionClosed(WsConnectionClosed):
    """Raised when a public method is called on a closed CDP connection."""

    def __init__(self, reason):
        """Constructor.

        :param reason:
        :type reason: wsproto.frame_protocol.CloseReason
        """
        self.reason = reason

    def __repr__(self):
        """Return representation."""
        return f'{self.__class__.__name__}<{self.reason}>'