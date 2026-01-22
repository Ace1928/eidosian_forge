from __future__ import annotations
import collections
import contextlib
import copy
import os
import platform
import socket
import ssl
import sys
import threading
import time
import weakref
from typing import (
import bson
from bson import DEFAULT_CODEC_OPTIONS
from bson.son import SON
from pymongo import __version__, _csot, auth, helpers
from pymongo.client_session import _validate_session_write_concern
from pymongo.common import (
from pymongo.errors import (
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_reauth
from pymongo.lock import _create_lock
from pymongo.monitoring import (
from pymongo.network import command, receive_message
from pymongo.read_preferences import ReadPreference
from pymongo.server_api import _add_to_command
from pymongo.server_type import SERVER_TYPE
from pymongo.socket_checker import SocketChecker
from pymongo.ssl_support import HAS_SNI, SSLError
class _PoolGeneration:

    def __init__(self) -> None:
        self._generations: dict[ObjectId, int] = collections.defaultdict(int)
        self._generation = 0

    def get(self, service_id: Optional[ObjectId]) -> int:
        """Get the generation for the given service_id."""
        if service_id is None:
            return self._generation
        return self._generations[service_id]

    def get_overall(self) -> int:
        """Get the Pool's overall generation."""
        return self._generation

    def inc(self, service_id: Optional[ObjectId]) -> None:
        """Increment the generation for the given service_id."""
        self._generation += 1
        if service_id is None:
            for service_id in self._generations:
                self._generations[service_id] += 1
        else:
            self._generations[service_id] += 1

    def stale(self, gen: int, service_id: Optional[ObjectId]) -> bool:
        """Return if the given generation for a given service_id is stale."""
        return gen != self.get(service_id)