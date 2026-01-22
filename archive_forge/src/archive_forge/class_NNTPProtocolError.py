import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
class NNTPProtocolError(NNTPError):
    """Response does not begin with [1-5]"""
    pass