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
def unjellyFor(self, unjellier, unjellyList):
    clsName, name, ref = unjellyList
    self.name = name
    if bytes != str and isinstance(self.name, bytes):
        self.name = self.name.decode('utf-8')
    return pb.RemoteReference.unjellyFor(self, unjellier, [clsName, ref])