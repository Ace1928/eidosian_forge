from email import encoders
from email.mime.nonmultipart import MIMENonMultipart
@rule
def _jpeg(h):
    """JPEG data with JFIF or Exif markers; and raw JPEG"""
    if h[6:10] in (b'JFIF', b'Exif'):
        return 'jpeg'
    elif h[:4] == b'\xff\xd8\xff\xdb':
        return 'jpeg'