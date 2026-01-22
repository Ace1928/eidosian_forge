import enum
import functools
import redis
from redis import exceptions as redis_exceptions
class UnknownExpire(enum.IntEnum):
    """Non-expiry (not ttls) results return from :func:`.get_expiry`.

    See: http://redis.io/commands/ttl or http://redis.io/commands/pttl
    """
    DOES_NOT_EXPIRE = -1
    '\n    The command returns ``-1`` if the key exists but has no associated expire.\n    '
    KEY_NOT_FOUND = -2