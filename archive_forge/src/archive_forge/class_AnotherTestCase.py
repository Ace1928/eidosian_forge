import base64
import inspect
import re
from io import BytesIO
from typing import Any, List, Optional, Tuple, Type
from zope.interface import directlyProvides, implementer
import twisted.cred.checkers
import twisted.cred.credentials
import twisted.cred.error
import twisted.cred.portal
from twisted import cred
from twisted.cred.checkers import AllowAnonymousAccess, ICredentialsChecker
from twisted.cred.credentials import IAnonymous
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import address, defer, error, interfaces, protocol, reactor, task
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.mail import smtp
from twisted.mail._cred import LOGINCredentials
from twisted.protocols import basic, loopback
from twisted.python.util import LineLog
from twisted.trial.unittest import TestCase
class AnotherTestCase:
    serverClass: Optional[Type[protocol.Protocol]] = None
    clientClass: Optional[Type[smtp.SMTPClient]] = None
    messages = [(b'foo.com', b'moshez@foo.com', [b'moshez@bar.com'], b'moshez@foo.com', [b'moshez@bar.com'], b'From: Moshe\nTo: Moshe\n\nHi,\nhow are you?\n'), (b'foo.com', b'tttt@rrr.com', [b'uuu@ooo', b'yyy@eee'], b'tttt@rrr.com', [b'uuu@ooo', b'yyy@eee'], b'Subject: pass\n\n..rrrr..\n'), (b'foo.com', b'@this,@is,@ignored:foo@bar.com', [b'@ignore,@this,@too:bar@foo.com'], b'foo@bar.com', [b'bar@foo.com'], b'Subject: apa\nTo: foo\n\n123\n.\n456\n')]
    data: List[Tuple[bytes, bytes, Any, Any]] = [(b'', b'220.*\r\n$', None, None), (b'HELO foo.com\r\n', b'250.*\r\n$', None, None), (b'RSET\r\n', b'250.*\r\n$', None, None)]
    for helo_, from_, to_, realfrom, realto, msg in messages:
        data.append((b'MAIL FROM:<' + from_ + b'>\r\n', b'250.*\r\n', None, None))
        for rcpt in to_:
            data.append((b'RCPT TO:<' + rcpt + b'>\r\n', b'250.*\r\n', None, None))
        data.append((b'DATA\r\n', b'354.*\r\n', msg, (b'250.*\r\n', (helo_, realfrom, realto, msg))))

    def test_buffer(self):
        """
        Exercise a lot of the SMTP client code.  This is a "shotgun" style unit
        test.  It does a lot of things and hopes that something will go really
        wrong if it is going to go wrong.  This test should be replaced with a
        suite of nicer tests.
        """
        transport = StringTransport()
        a = self.serverClass()

        class fooFactory:
            domain = b'foo.com'
        a.factory = fooFactory()
        a.makeConnection(transport)
        for send, expect, msg, msgexpect in self.data:
            if send:
                a.dataReceived(send)
            data = transport.value()
            transport.clear()
            if not re.match(expect, data):
                raise AssertionError(send, expect, data)
            if data[:3] == b'354':
                for line in msg.splitlines():
                    if line and line[0:1] == b'.':
                        line = b'.' + line
                    a.dataReceived(line + b'\r\n')
                a.dataReceived(b'.\r\n')
                data = transport.value()
                transport.clear()
                resp, msgdata = msgexpect
                if not re.match(resp, data):
                    raise AssertionError(resp, data)
                for recip in msgdata[2]:
                    expected = list(msgdata[:])
                    expected[2] = [recip]
                    self.assertEqual(a.message[recip,], tuple(expected))
        a.setTimeout(None)