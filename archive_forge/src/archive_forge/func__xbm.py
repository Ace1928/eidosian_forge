from email import encoders
from email.mime.nonmultipart import MIMENonMultipart
@rule
def _xbm(h):
    """X bitmap (X10 or X11)"""
    if h.startswith(b'#define '):
        return 'xbm'