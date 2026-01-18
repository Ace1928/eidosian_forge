import sys
from email.header import decode_header
from .. import __version__ as _breezy_version
from .. import tests
from ..email_message import EmailMessage
from ..errors import BzrBadParameterNotUnicode
from ..smtp_connection import SMTPConnection
def test_send_plain(self):
    self.send_email('a\nb\nc\nd\ne\n', 'lines.txt')
    self.assertMessage(complex_multipart_message('plain'))