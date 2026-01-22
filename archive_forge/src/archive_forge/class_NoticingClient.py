import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
class NoticingClient(IRCClientWithoutLogin):
    methods = {'created': ('when',), 'yourHost': ('info',), 'myInfo': ('servername', 'version', 'umodes', 'cmodes'), 'luserClient': ('info',), 'bounce': ('info',), 'isupport': ('options',), 'luserChannels': ('channels',), 'luserOp': ('ops',), 'luserMe': ('info',), 'receivedMOTD': ('motd',), 'privmsg': ('user', 'channel', 'message'), 'joined': ('channel',), 'left': ('channel',), 'noticed': ('user', 'channel', 'message'), 'modeChanged': ('user', 'channel', 'set', 'modes', 'args'), 'pong': ('user', 'secs'), 'signedOn': (), 'kickedFrom': ('channel', 'kicker', 'message'), 'nickChanged': ('nick',), 'userJoined': ('user', 'channel'), 'userLeft': ('user', 'channel'), 'userKicked': ('user', 'channel', 'kicker', 'message'), 'action': ('user', 'channel', 'data'), 'topicUpdated': ('user', 'channel', 'newTopic'), 'userRenamed': ('oldname', 'newname')}

    def __init__(self, *a, **kw):
        self.calls = []

    def __getattribute__(self, name):
        if name.startswith('__') and name.endswith('__'):
            return super().__getattribute__(name)
        try:
            args = super().__getattribute__('methods')[name]
        except KeyError:
            return super().__getattribute__(name)
        else:
            return self.makeMethod(name, args)

    def makeMethod(self, fname, args):

        def method(*a, **kw):
            if len(a) > len(args):
                raise TypeError('TypeError: %s() takes %d arguments (%d given)' % (fname, len(args), len(a)))
            for name, value in zip(args, a):
                if name in kw:
                    raise TypeError("TypeError: %s() got multiple values for keyword argument '%s'" % (fname, name))
                else:
                    kw[name] = value
            if len(kw) != len(args):
                raise TypeError('TypeError: %s() takes %d arguments (%d given)' % (fname, len(args), len(a)))
            self.calls.append((fname, kw))
        return method