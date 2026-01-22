import base64
import binascii
import re
import string
import six
class Base64ValueError(Exception):
    """Illegal Base64-encoded value"""