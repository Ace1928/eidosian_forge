from email import encoders
from email.mime.nonmultipart import MIMENonMultipart
@rule
def _pgm(h):
    """PGM (portable graymap)"""
    if len(h) >= 3 and h[0] == ord(b'P') and (h[1] in b'25') and (h[2] in b' \t\n\r'):
        return 'pgm'