import numbers
import os
import sys
from datetime import timedelta
from typing import Dict, Optional
from torch.distributed import FileStore, PrefixStore, Store, TCPStore
from .constants import default_pg_timeout
def _create_c10d_store(hostname, port, rank, world_size, timeout, use_libuv=False) -> Store:
    """
    Smartly creates a c10d Store object on ``rank`` based on whether we need to re-use agent store.

    The TCPStore server is assumed to be hosted
    on ``hostname:port``.

    If ``torchelastic_use_agent_store()`` is ``True``, then it is assumed that
    the agent leader (node rank 0) hosts the TCPStore server (for which the
    endpoint is specified by the given ``hostname:port``). Hence
    ALL ranks will create and return a TCPStore client (e.g. ``start_daemon=False``).

    If ``torchelastic_use_agent_store()`` is ``False``, then rank 0 will host
    the TCPStore (with multi-tenancy) and it is assumed that rank 0's hostname
    and port are correctly passed via ``hostname`` and ``port``. All
    non-zero ranks will create and return a TCPStore client.
    """
    if not 0 <= port < 2 ** 16:
        raise ValueError(f'port must have value from 0 to 65535 but was {port}.')
    if _torchelastic_use_agent_store():
        attempt = os.environ['TORCHELASTIC_RESTART_COUNT']
        tcp_store = TCPStore(hostname, port, world_size, False, timeout)
        return PrefixStore(f'/worker/attempt_{attempt}', tcp_store)
    else:
        start_daemon = rank == 0
        return TCPStore(hostname, port, world_size, start_daemon, timeout, multi_tenant=True, use_libuv=use_libuv)