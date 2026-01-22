from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
class RimuHostingException(Exception):
    """
    Exception class for RimuHosting driver
    """

    def __str__(self):
        return self.args[0]

    def __repr__(self):
        return "<RimuHostingException '%s'>" % self.args[0]