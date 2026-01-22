import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
class OpenNebulaNodeSize(NodeSize):
    """
    NodeSize class for the OpenNebula.org driver.
    """

    def __init__(self, id, name, ram, disk, bandwidth, price, driver, cpu=None, vcpu=None):
        super().__init__(id=id, name=name, ram=ram, disk=disk, bandwidth=bandwidth, price=price, driver=driver)
        self.cpu = cpu
        self.vcpu = vcpu

    def __repr__(self):
        return '<OpenNebulaNodeSize: id=%s, name=%s, ram=%s, disk=%s, bandwidth=%s, price=%s, driver=%s, cpu=%s, vcpu=%s ...>' % (self.id, self.name, self.ram, self.disk, self.bandwidth, self.price, self.driver.name, self.cpu, self.vcpu)