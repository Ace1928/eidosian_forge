import random
import threading
import time
import typing
from typing import Any
from typing import Mapping
import warnings
from ..api import CacheBackend
from ..api import NO_VALUE
from ... import util
class ClientPool(threading.local):

    def __init__(self):
        self.memcached = backend._create_client()