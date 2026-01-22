import struct
import typing as t
def GG(a: int, b: int, c: int, d: int, x: int, s: int) -> int:
    return ROTL(a + G(b, c, d) + x + 1518500249 & 4294967295, s)