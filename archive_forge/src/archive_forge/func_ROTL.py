import struct
import typing as t
def ROTL(x: int, n: int) -> int:
    return x << n & 4294967295 | x >> 32 - n