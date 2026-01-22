import errno
import smtplib
import socket
from email.utils import getaddresses, parseaddr
from . import config, osutils
from .errors import BzrError, InternalBzrError
class SMTPConnectionRefused(SMTPError):
    _fmt = 'SMTP connection to %(host)s refused'

    def __init__(self, error, host):
        self.error = error
        self.host = host