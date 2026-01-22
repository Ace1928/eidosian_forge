import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
class MacAddressAlreadyExist(os_ken_exc.OSKenException):
    message = 'port (%(dpid)s, %(port)s) has already mac %(mac_address)s'