from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.internet.defer import succeed
from twisted.words.im import basesupport, interfaces, locals
from twisted.words.im.locals import ONLINE
from twisted.words.protocols import irc

        RPL_NAMREPLY
        >> NAMES #bnl
        << :Arlington.VA.US.Undernet.Org 353 z3p = #bnl :pSwede Dan-- SkOyg AG
        