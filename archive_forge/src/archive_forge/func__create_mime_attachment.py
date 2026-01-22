import mimetypes
from email import charset as Charset
from email import encoders as Encoders
from email import generator, message_from_string
from email.errors import HeaderParseError
from email.header import Header
from email.headerregistry import Address, parser
from email.message import Message
from email.mime.base import MIMEBase
from email.mime.message import MIMEMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, getaddresses, make_msgid
from io import BytesIO, StringIO
from pathlib import Path
from django.conf import settings
from django.core.mail.utils import DNS_NAME
from django.utils.encoding import force_str, punycode
def _create_mime_attachment(self, content, mimetype):
    """
        Convert the content, mimetype pair into a MIME attachment object.

        If the mimetype is message/rfc822, content may be an
        email.Message or EmailMessage object, as well as a str.
        """
    basetype, subtype = mimetype.split('/', 1)
    if basetype == 'text':
        encoding = self.encoding or settings.DEFAULT_CHARSET
        attachment = SafeMIMEText(content, subtype, encoding)
    elif basetype == 'message' and subtype == 'rfc822':
        if isinstance(content, EmailMessage):
            content = content.message()
        elif not isinstance(content, Message):
            content = message_from_string(force_str(content))
        attachment = SafeMIMEMessage(content, subtype)
    else:
        attachment = MIMEBase(basetype, subtype)
        attachment.set_payload(content)
        Encoders.encode_base64(attachment)
    return attachment