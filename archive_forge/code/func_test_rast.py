from os import PathLike
import warnings
def test_rast(h, f):
    """Sun raster file"""
    if h.startswith(b'Y\xa6j\x95'):
        return 'rast'