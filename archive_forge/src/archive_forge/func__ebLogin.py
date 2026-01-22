from time import ctime, time
from zope.interface import implementer
from twisted import copyright
from twisted.cred import credentials, error as ecred, portal
from twisted.internet import defer, protocol
from twisted.python import failure, log, reflect
from twisted.python.components import registerAdapter
from twisted.spread import pb
from twisted.words import ewords, iwords
from twisted.words.protocols import irc
def _ebLogin(self, err, nickname):
    if err.check(ewords.AlreadyLoggedIn):
        self.privmsg(NICKSERV, nickname, 'Already logged in.  No pod people allowed!')
    elif err.check(ecred.UnauthorizedLogin):
        self.privmsg(NICKSERV, nickname, 'Login failed.  Goodbye.')
    else:
        log.msg('Unhandled error during login:')
        log.err(err)
        self.privmsg(NICKSERV, nickname, 'Server error during login.  Sorry.')
    self.transport.loseConnection()