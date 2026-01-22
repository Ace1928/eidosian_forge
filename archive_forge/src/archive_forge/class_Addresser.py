import abc
import logging
from oslo_messaging.target import Target
class Addresser(object):
    """Base class message bus address generator. Used to convert an
    oslo.messaging address into an AMQP 1.0 address string used over the
    connection to the message bus.
    """

    def __init__(self, default_exchange):
        self._default_exchange = default_exchange

    def resolve(self, target, service):
        if not isinstance(target, Target):
            return target
        if target.fanout:
            return self.multicast_address(target, service)
        elif target.server:
            return self.unicast_address(target, service)
        else:
            return self.anycast_address(target, service)

    @abc.abstractmethod
    def multicast_address(self, target, service):
        """Address used to broadcast to all subscribers
        """

    @abc.abstractmethod
    def unicast_address(self, target, service):
        """Address used to target a specific subscriber (direct)
        """

    @abc.abstractmethod
    def anycast_address(self, target, service):
        """Address used for shared subscribers (competing consumers)
        """

    def _concat(self, sep, items):
        return sep.join(filter(bool, items))