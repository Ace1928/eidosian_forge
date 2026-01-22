import errno
import smtplib
import socket
from email.utils import getaddresses, parseaddr
from . import config, osutils
from .errors import BzrError, InternalBzrError
class NoDestinationAddress(InternalBzrError):
    _fmt = 'Message does not have a destination address.'