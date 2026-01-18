from os import PathLike
import warnings
def test_webp(h, f):
    if h.startswith(b'RIFF') and h[8:12] == b'WEBP':
        return 'webp'