import snappy
import FXrays
def weak_normalize_slope(slope):
    """For a tuple (a, b), scale it so that gcd(a,b)=1"""
    a, b = [int(s) for s in slope]
    if a == b == 0:
        return (0, 0)
    g = gcd(a, b)
    a, b = (a // g, b // g)
    return (a, b)