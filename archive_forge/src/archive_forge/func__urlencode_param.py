import functools
from enum import Enum
from urllib.parse import urlencode
@functools.lru_cache(maxsize=ROUTING_PARAM_CACHE_SIZE)
def _urlencode_param(key, value):
    """Cacheable wrapper over urlencode

    Args:
        key (str): The key of the parameter to encode.
        value (str | bytes | Enum): The value of the parameter to encode.

    Returns:
        str: The encoded parameter.
    """
    return urlencode({key: value}, safe='/')