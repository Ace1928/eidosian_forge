from __future__ import annotations
import contextlib
import enum
import errno
import hmac
import os
import struct
import warnings
import weakref
from itertools import count
from typing import (
from weakref import ReferenceType, WeakValueDictionary
import attrs
import trio
from ._util import NoPublicConstructor, final
Initiate an outgoing DTLS connection.

        Notice that this is a synchronous method. That's because it doesn't actually
        initiate any I/O – it just sets up a `DTLSChannel` object. The actual handshake
        doesn't occur until you start using the `DTLSChannel`. This gives you a chance
        to do further configuration first, like setting MTU etc.

        Args:
          address: The address to connect to. Usually a (host, port) tuple, like
            ``("127.0.0.1", 12345)``.
          ssl_context (OpenSSL.SSL.Context): The PyOpenSSL context object to use for
            this connection.

        Returns:
          DTLSChannel

        