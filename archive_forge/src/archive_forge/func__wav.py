from io import BytesIO
from email import encoders
from email.mime.nonmultipart import MIMENonMultipart
@rule
def _wav(h, f):
    if not h.startswith(b'RIFF') or h[8:12] != b'WAVE' or h[12:16] != b'fmt ':
        return None
    else:
        return 'x-wav'