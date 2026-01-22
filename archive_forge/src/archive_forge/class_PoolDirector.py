import hashlib
import io
import json
import os
import platform
import random
import socket
import ssl
import threading
import time
import urllib.parse
from typing import (
import filelock
import urllib3
from blobfile import _xml as xml
class PoolDirector:

    def __init__(self, connection_pool_max_size: int, max_connection_pool_count: int) -> None:
        self.connection_pool_max_size = connection_pool_max_size
        self.max_connection_pool_count = max_connection_pool_count
        self.pool_manager = None
        self.creation_pid = None
        self.lock = threading.Lock()

    def get_http_pool(self) -> urllib3.PoolManager:
        with self.lock:
            if self.pool_manager is None or self.creation_pid != os.getpid():
                self.creation_pid = os.getpid()
                self.pool_manager = urllib3.PoolManager(maxsize=self.connection_pool_max_size, num_pools=self.max_connection_pool_count)
            return self.pool_manager

    def __getstate__(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k not in ['lock', 'pool_manager']}

    def __setstate__(self, state: Any) -> None:
        self.__init__(connection_pool_max_size=state['connection_pool_max_size'], max_connection_pool_count=state['max_connection_pool_count'])
        self.__dict__.update(state)