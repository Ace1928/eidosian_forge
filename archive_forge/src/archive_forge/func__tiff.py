from email import encoders
from email.mime.nonmultipart import MIMENonMultipart
@rule
def _tiff(h):
    """TIFF (can be in Motorola or Intel byte order)"""
    if h[:2] in (b'MM', b'II'):
        return 'tiff'