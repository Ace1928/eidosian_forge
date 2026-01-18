import sys
from email.header import decode_header
from .. import __version__ as _breezy_version
from .. import tests
from ..email_message import EmailMessage
from ..errors import BzrBadParameterNotUnicode
from ..smtp_connection import SMTPConnection
def test_retrieving_headers(self):
    msg = EmailMessage('from@from.com', 'to@to.com', 'subject')
    for header, value in [('From', 'from@from.com'), ('To', 'to@to.com'), ('Subject', 'subject')]:
        self.assertEqual(value, msg.get(header))
        self.assertEqual(value, msg[header])
    self.assertEqual(None, msg.get('Does-Not-Exist'))
    self.assertEqual(None, msg['Does-Not-Exist'])
    self.assertEqual('None', msg.get('Does-Not-Exist', 'None'))