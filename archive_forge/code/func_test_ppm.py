from os import PathLike
import warnings
def test_ppm(h, f):
    """PPM (portable pixmap)"""
    if len(h) >= 3 and h[0] == ord(b'P') and (h[1] in b'36') and (h[2] in b' \t\n\r'):
        return 'ppm'