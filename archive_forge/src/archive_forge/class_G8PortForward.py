import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
class G8PortForward(UuidMixin):

    def __init__(self, network, node_id, publicport, privateport, protocol, driver):
        self.node_id = node_id
        self.network = network
        self.publicport = int(publicport)
        self.privateport = int(privateport)
        self.protocol = protocol
        self.driver = driver
        UuidMixin.__init__(self)

    def destroy(self):
        self.driver.ex_delete_portforward(self)