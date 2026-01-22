from __future__ import annotations
import email.message
import email.parser
from io import BytesIO, StringIO
from typing import IO, AnyStr, Callable
from twisted.mail import bounce
from twisted.trial import unittest
def _bounceBigMessage(self, header: AnyStr, message: AnyStr, ioType: Callable[[AnyStr], IO[AnyStr]]) -> None:
    """
        Pass a really big message to L{twisted.mail.bounce.generateBounce}.
        """
    fromAddress, to, s = bounce.generateBounce(ioType(header + message), 'moshez@example.com', 'nonexistent@example.org')
    emailParser = email.parser.Parser()
    mess = emailParser.parse(StringIO(s.decode('utf-8')))
    self.assertEqual(mess['To'], 'moshez@example.com')
    self.assertEqual(mess['From'], 'postmaster@example.org')
    self.assertEqual(mess['subject'], 'Returned Mail: see transcript for details')
    self.assertTrue(mess.is_multipart())
    parts = mess.get_payload()
    innerMessage = parts[1].get_payload()
    if isinstance(message, bytes):
        messageText = message.decode('utf-8')
    else:
        messageText = message
    self.assertEqual(innerMessage[0].get_payload() + '\n', messageText)