import numbers
import os
import sys
from datetime import timedelta
from typing import Dict, Optional
from torch.distributed import FileStore, PrefixStore, Store, TCPStore
from .constants import default_pg_timeout
def _file_rendezvous_handler(url: str, **kwargs):

    def _error(msg):
        return _rendezvous_error('file:// rendezvous: ' + msg)
    result = urlparse(url)
    path = result.path
    if sys.platform == 'win32':
        import urllib.request
        full_path = result.netloc + result.path
        path = urllib.request.url2pathname(full_path)
        if path:
            path = os.path.normpath(path)
    if not path:
        raise _error('path missing')
    query_dict = _query_to_dict(result.query)
    if 'rank' not in query_dict:
        raise _error('rank parameter missing')
    if 'world_size' not in query_dict:
        raise _error('world size parameter missing')
    rank = int(query_dict['rank'])
    world_size = int(query_dict['world_size'])
    store = FileStore(path, world_size)
    yield (store, rank, world_size)
    raise RuntimeError('Unable to perform rerendezvous using file:// method')