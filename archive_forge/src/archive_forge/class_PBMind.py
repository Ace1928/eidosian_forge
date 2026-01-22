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
class PBMind(pb.Referenceable):

    def __init__(self):
        pass

    def jellyFor(self, jellier):
        qual = reflect.qual(PBMind)
        if isinstance(qual, str):
            qual = qual.encode('utf-8')
        return (qual, jellier.invoker.registerReference(self))

    def remote_userJoined(self, user, group):
        pass

    def remote_userLeft(self, user, group, reason):
        pass

    def remote_receive(self, sender, recipient, message):
        pass

    def remote_groupMetaUpdate(self, group, meta):
        pass