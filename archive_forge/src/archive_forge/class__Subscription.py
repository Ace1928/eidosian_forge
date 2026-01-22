import collections
from grpc.framework.interfaces.base import base
class _Subscription(base.Subscription, collections.namedtuple('_Subscription', ('kind', 'termination_callback', 'allowance', 'operator', 'protocol_receiver'))):
    """A trivial implementation of base.Subscription."""