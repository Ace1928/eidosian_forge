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
def _create_file_store(params: RendezvousParameters) -> FileStore:
    if params.endpoint:
        path = params.endpoint
    else:
        try:
            _, path = tempfile.mkstemp()
        except OSError as exc:
            raise RendezvousError('The file creation for C10d store has failed. See inner exception for details.') from exc
    try:
        store = FileStore(path)
    except (ValueError, RuntimeError) as exc:
        raise RendezvousConnectionError('The connection to the C10d store has failed. See inner exception for details.') from exc
    return store