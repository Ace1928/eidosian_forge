from email import encoders
from email.mime.nonmultipart import MIMENonMultipart
@rule
def _bmp(h):
    if h.startswith(b'BM'):
        return 'bmp'