import errno
import smtplib
import socket
from email.utils import getaddresses, parseaddr
from . import config, osutils
from .errors import BzrError, InternalBzrError
class DefaultSMTPConnectionRefused(SMTPConnectionRefused):
    _fmt = 'Please specify smtp_server.  No server at default %(host)s.'