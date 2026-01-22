import time
import hashlib
from libcloud.utils.py3 import b
from libcloud.common.base import ConnectionKey
from libcloud.common.xmlrpc import XMLRPCResponse, XMLRPCConnection
class GandiException(Exception):
    """
    Exception class for Gandi driver
    """

    def __str__(self):
        return '({}) {}'.format(self.args[0], self.args[1])

    def __repr__(self):
        return '<GandiException code {} "{}">'.format(self.args[0], self.args[1])