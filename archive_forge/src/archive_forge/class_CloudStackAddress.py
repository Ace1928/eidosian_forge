import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackAddress:
    """
    A public IP address.

    :param      id: UUID of the Public IP
    :type       id: ``str``

    :param      address: The public IP address
    :type       address: ``str``

    :param      associated_network_id: The ID of the network where this address
                                        has been associated with
    :type       associated_network_id: ``str``

    :param      vpc_id: VPC the ip belongs to
    :type       vpc_id: ``str``

    :param      virtualmachine_id: The ID of virtual machine this address
                                   is assigned to
    :type       virtualmachine_id: ``str``
    """

    def __init__(self, id, address, driver, associated_network_id=None, vpc_id=None, virtualmachine_id=None):
        self.id = id
        self.address = address
        self.driver = driver
        self.associated_network_id = associated_network_id
        self.vpc_id = vpc_id
        self.virtualmachine_id = virtualmachine_id

    def release(self):
        self.driver.ex_release_public_ip(address=self)

    def __str__(self):
        return self.address

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.id == other.id