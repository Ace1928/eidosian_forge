import random
from hashlib import md5
from zope.interface import Interface, implementer
from twisted.cred.credentials import (
from twisted.cred.portal import Portal
from twisted.internet import defer, protocol
from twisted.persisted import styles
from twisted.python import failure, log, reflect
from twisted.python.compat import cmp, comparable
from twisted.python.components import registerAdapter
from twisted.spread import banana
from twisted.spread.flavors import (
from twisted.spread.interfaces import IJellyable, IUnjellyable
from twisted.spread.jelly import _newInstance, globalSecurity, jelly, unjelly
class IPerspective(Interface):
    """
    per*spec*tive, n. : The relationship of aspects of a subject to each
    other and to a whole: 'a perspective of history'; 'a need to view
    the problem in the proper perspective'.

    This is a Perspective Broker-specific wrapper for an avatar. That
    is to say, a PB-published view on to the business logic for the
    system's concept of a 'user'.

    The concept of attached/detached is no longer implemented by the
    framework. The realm is expected to implement such semantics if
    needed.
    """

    def perspectiveMessageReceived(broker, message, args, kwargs):
        """
        This method is called when a network message is received.

        @arg broker: The Perspective Broker.

        @type message: str
        @arg message: The name of the method called by the other end.

        @type args: list in jelly format
        @arg args: The arguments that were passed by the other end. It
                   is recommend that you use the `unserialize' method of the
                   broker to decode this.

        @type kwargs: dict in jelly format
        @arg kwargs: The keyword arguments that were passed by the
                     other end.  It is recommended that you use the
                     `unserialize' method of the broker to decode this.

        @rtype: A jelly list.
        @return: It is recommended that you use the `serialize' method
                 of the broker on whatever object you need to return to
                 generate the return value.
        """