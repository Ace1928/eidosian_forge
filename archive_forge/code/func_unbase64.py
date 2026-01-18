from base64 import b64encode, b64decode
import binascii
def unbase64(s: Base64String) -> str:
    """Decode the string s using Base64."""
    try:
        b: bytes = s.encode('ascii') if isinstance(s, str) else s
    except UnicodeEncodeError:
        return ''
    try:
        return b64decode(b).decode('utf-8')
    except (binascii.Error, UnicodeDecodeError):
        return ''