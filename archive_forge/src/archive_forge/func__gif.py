from email import encoders
from email.mime.nonmultipart import MIMENonMultipart
@rule
def _gif(h):
    """GIF ('87 and '89 variants)"""
    if h[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'