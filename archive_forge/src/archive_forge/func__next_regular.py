from packaging.version import Version, parse
import numpy as np
import scipy
def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    Target must be a positive integer.
    """
    if target <= 6:
        return target
    if not target & target - 1:
        return target
    match = float('inf')
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            quotient = -(-target // p35)
            p2 = 2 ** (quotient - 1).bit_length()
            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match