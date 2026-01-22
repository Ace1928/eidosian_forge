from abc import ABCMeta
from abc import abstractmethod
import logging
import sys
import threading
from oslo_config import cfg
from oslo_utils import eventletutils
from oslo_messaging import _utils as utils
from oslo_messaging import dispatcher
from oslo_messaging import serializer as msg_serializer
from oslo_messaging import server as msg_server
from oslo_messaging import target as msg_target
class RPCAccessPolicyBase(object, metaclass=ABCMeta):
    """Determines which endpoint methods may be invoked via RPC"""

    @abstractmethod
    def is_allowed(self, endpoint, method):
        """Applies an access policy to the rpc method
        :param endpoint: the instance of a rpc endpoint
        :param method: the method of the endpoint
        :return: True if the method may be invoked via RPC, else False.
        """