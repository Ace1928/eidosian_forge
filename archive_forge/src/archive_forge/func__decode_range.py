import bisect
def _decode_range(r):
    return (r >> 32, r & (1 << 32) - 1)