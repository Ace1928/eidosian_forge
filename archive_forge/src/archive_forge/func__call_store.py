import binascii
import logging
import os
import tempfile
from base64 import b64decode, b64encode
from datetime import timedelta
from typing import Any, Optional, Tuple, cast
from torch.distributed import FileStore, Store, TCPStore
from torch.distributed.elastic.events import (
from .api import (
from .dynamic_rendezvous import RendezvousBackend, Token
from .utils import _matches_machine_hostname, parse_rendezvous_endpoint
def _call_store(self, store_op: str, *args, **kwargs) -> Any:
    try:
        return getattr(self._store, store_op)(*args, **kwargs)
    except (ValueError, RuntimeError, TimeoutError) as exc:
        raise RendezvousConnectionError('The connection to the C10d store has failed. See inner exception for details.') from exc