import warnings
from contextlib import contextmanager
from typing import Optional, Tuple, Type
from redis import Connection as RedisConnection
from redis import Redis
from .local import LocalStack
class NoRedisConnectionException(Exception):
    pass