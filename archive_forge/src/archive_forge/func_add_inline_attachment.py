from email.header import Header
from email.message import Message
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr
from . import __version__ as _breezy_version
from .errors import BzrBadParameterNotUnicode
from .osutils import safe_unicode
from .smtp_connection import SMTPConnection
def add_inline_attachment(self, body, filename=None, mime_subtype='plain'):
    """Add an inline attachment to the message.

        :param body: A text to attach. Can be an unicode string or a byte
            string, and it'll be sent as ascii, utf-8, or 8-bit, in that
            preferred order.
        :param filename: The name for the attachment. This will give a default
            name for email programs to save the attachment.
        :param mime_subtype: MIME subtype of the attachment (eg. 'plain' for
            text/plain [default]).

        The attachment body will be displayed inline, so do not use this
        function to attach binary attachments.
        """
    if self._body is not None:
        self._parts.append((self._body, None, 'plain'))
        self._body = None
    self._parts.append((body, filename, mime_subtype))