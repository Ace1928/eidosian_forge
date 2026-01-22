import base64
from hashlib import sha256
import hmac
import binascii
from six import text_type, binary_type
def add_base64_padding(b):
    """Add padding to base64 encoded bytes.

    Padding can be removed when sending the messages.

    @param b bytes to be padded.
    @return a padded bytes.
    """
    return b + b'=' * (-len(b) % 4)