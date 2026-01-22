import random
import socket
import netaddr
from neutron_lib import constants
class _AuthenticBase(object):

    def __init__(self, addr, **kwargs):
        super().__init__(addr, **kwargs)
        self._initial_value = addr

    def __str__(self):
        if isinstance(self._initial_value, str):
            return self._initial_value
        return super().__str__()

    def __deepcopy__(self, memo):
        return self.__class__(self._initial_value)