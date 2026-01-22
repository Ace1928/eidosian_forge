from email import encoders
from email.mime.nonmultipart import MIMENonMultipart
@rule
def _webp(h):
    if h.startswith(b'RIFF') and h[8:12] == b'WEBP':
        return 'webp'