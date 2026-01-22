import numbers
import os
import sys
from datetime import timedelta
from typing import Dict, Optional
from torch.distributed import FileStore, PrefixStore, Store, TCPStore
from .constants import default_pg_timeout
def _create_store_from_options(backend_options, rank):
    store, _, _ = next(_rendezvous_helper(backend_options.init_method, rank, None))
    return store