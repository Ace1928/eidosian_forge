import os
import zlib
import time  # noqa
import logging
import numpy as np
def twits2bits(arr):
    """Given a few (signed) numbers, store them
    as compactly as possible in the wat specifief by the swf format.
    The numbers are multiplied by 20, assuming they
    are twits.
    Can be used to make the RECT record.
    """
    maxlen = 1
    for i in arr:
        tmp = len(signedint2bits(i * 20))
        if tmp > maxlen:
            maxlen = tmp
    bits = int2bits(maxlen, 5)
    for i in arr:
        bits += signedint2bits(i * 20, maxlen)
    return bits