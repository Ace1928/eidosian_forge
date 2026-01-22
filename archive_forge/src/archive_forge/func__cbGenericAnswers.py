import struct
from twisted.conch import error, interfaces
from twisted.conch.ssh import keys, service, transport
from twisted.conch.ssh.common import NS, getNS
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, reactor
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString
def _cbGenericAnswers(self, responses):
    """
        Called back when we are finished answering keyboard-interactive
        questions.  Send the info back to the server in a
        MSG_USERAUTH_INFO_RESPONSE.

        @param responses: a list of L{bytes} responses
        @type responses: L{list}
        """
    data = struct.pack('!L', len(responses))
    for r in responses:
        data += NS(r.encode('UTF8'))
    self.transport.sendPacket(MSG_USERAUTH_INFO_RESPONSE, data)