import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def test_read_segments_lock():
    fobj = BytesIO()
    arr = np.array(np.random.randint(0, 256, 1000), dtype=np.uint8)
    fobj.write(arr.tobytes())

    def yielding_read(*args, **kwargs):
        time.sleep(0.001)
        return fobj._real_read(*args, **kwargs)
    fobj._real_read = fobj.read
    fobj.read = yielding_read

    def random_segments(nsegs):
        segs = []
        nbytes = 0
        for i in range(nsegs):
            seglo = np.random.randint(0, 998)
            seghi = np.random.randint(seglo + 1, 1000)
            seglen = seghi - seglo
            nbytes += seglen
            segs.append([seglo, seglen])
        return (segs, nbytes)

    def get_expected(segs):
        segs = [arr[off:off + length] for off, length in segs]
        return np.concatenate(segs)
    numpassed = [0]
    lock = Lock()

    def runtest():
        seg, nbytes = random_segments(1)
        expected = get_expected(seg)
        _check_bytes(read_segments(fobj, seg, nbytes, lock), expected)
        seg, nbytes = random_segments(10)
        expected = get_expected(seg)
        _check_bytes(read_segments(fobj, seg, nbytes, lock), expected)
        with lock:
            numpassed[0] += 1
    threads = [Thread(target=runtest) for i in range(100)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert numpassed[0] == len(threads)