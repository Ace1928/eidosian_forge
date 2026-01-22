from email.header import Header
from email.message import Message
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr
from . import __version__ as _breezy_version
from .errors import BzrBadParameterNotUnicode
from .osutils import safe_unicode
from .smtp_connection import SMTPConnection
Return a str object together with an encoding.

        :param string\_: A str or unicode object.
        :return: A tuple (str, encoding), where encoding is one of 'ascii',
            'utf-8', or '8-bit', in that preferred order.
        