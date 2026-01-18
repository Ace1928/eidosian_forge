import os
import sys
from enum import Enum, _simple_enum
def uuid3(namespace, name):
    """Generate a UUID from the MD5 hash of a namespace UUID and a name."""
    from hashlib import md5
    digest = md5(namespace.bytes + bytes(name, 'utf-8'), usedforsecurity=False).digest()
    return UUID(bytes=digest[:16], version=3)