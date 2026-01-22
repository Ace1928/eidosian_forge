import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
class TestMind(service.PBMind):

    def __init__(self, *a, **kw):
        self.joins = []
        self.parts = []
        self.messages = []
        self.meta = []

    def remote_userJoined(self, user, group):
        self.joins.append((user, group))

    def remote_userLeft(self, user, group, reason):
        self.parts.append((user, group, reason))

    def remote_receive(self, sender, recipient, message):
        self.messages.append((sender, recipient, message))

    def remote_groupMetaUpdate(self, group, meta):
        self.meta.append((group, meta))