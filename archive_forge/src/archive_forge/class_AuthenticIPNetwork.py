import random
import socket
import netaddr
from neutron_lib import constants
class AuthenticIPNetwork(_AuthenticBase, netaddr.IPNetwork):
    """AuthenticIPNetwork class

    This class retains the format of the IP network string passed during
    initialization.

    This is useful when we want to make sure that we retain the format passed
    by a user through API.
    """