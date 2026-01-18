import os
import sys
from enum import Enum, _simple_enum
def uuid1(node=None, clock_seq=None):
    """Generate a UUID from a host ID, sequence number, and the current time.
    If 'node' is not given, getnode() is used to obtain the hardware
    address.  If 'clock_seq' is given, it is used as the sequence number;
    otherwise a random 14-bit sequence number is chosen."""
    if _generate_time_safe is not None and node is clock_seq is None:
        uuid_time, safely_generated = _generate_time_safe()
        try:
            is_safe = SafeUUID(safely_generated)
        except ValueError:
            is_safe = SafeUUID.unknown
        return UUID(bytes=uuid_time, is_safe=is_safe)
    global _last_timestamp
    import time
    nanoseconds = time.time_ns()
    timestamp = nanoseconds // 100 + 122192928000000000
    if _last_timestamp is not None and timestamp <= _last_timestamp:
        timestamp = _last_timestamp + 1
    _last_timestamp = timestamp
    if clock_seq is None:
        import random
        clock_seq = random.getrandbits(14)
    time_low = timestamp & 4294967295
    time_mid = timestamp >> 32 & 65535
    time_hi_version = timestamp >> 48 & 4095
    clock_seq_low = clock_seq & 255
    clock_seq_hi_variant = clock_seq >> 8 & 63
    if node is None:
        node = getnode()
    return UUID(fields=(time_low, time_mid, time_hi_version, clock_seq_hi_variant, clock_seq_low, node), version=1)