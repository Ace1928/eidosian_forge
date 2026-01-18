import os
import re
import copy
import time
import base64
import datetime
from xml.parsers.expat import ExpatError
from libcloud.utils.py3 import ET, b, next, httplib, urlparse, urlencode
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
@classmethod
def to_lease(cls, lease_element):
    """
        Convert lease settings element to lease instance.

        :param lease_element: "LeaseSettingsSection" XML element
        :type lease_element: ``ET.Element``

        :return: Lease instance
        :rtype: :class:`Lease`
        """
    lease_id = lease_element.get('href')
    deployment_lease = lease_element.find(fixxpath(lease_element, 'DeploymentLeaseInSeconds'))
    storage_lease = lease_element.find(fixxpath(lease_element, 'StorageLeaseInSeconds'))
    deployment_lease_expiration = lease_element.find(fixxpath(lease_element, 'DeploymentLeaseExpiration'))
    storage_lease_expiration = lease_element.find(fixxpath(lease_element, 'StorageLeaseExpiration'))

    def apply_if_elem_not_none(elem, function):
        return function(elem.text) if elem is not None else None
    return cls(lease_id=lease_id, deployment_lease=apply_if_elem_not_none(deployment_lease, int), storage_lease=apply_if_elem_not_none(storage_lease, int), deployment_lease_expiration=apply_if_elem_not_none(deployment_lease_expiration, parse_date), storage_lease_expiration=apply_if_elem_not_none(storage_lease_expiration, parse_date))