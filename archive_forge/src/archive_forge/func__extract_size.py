import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
def _extract_size(self, compute):
    """
        Extract size, or node type, from a compute node XML representation.

        Extract node size, or node type, description from a compute node XML
        representation, converting the node size to a NodeSize object.

        :type  compute: :class:`ElementTree`
        :param compute: XML representation of a compute node.

        :rtype:  :class:`OpenNebulaNodeSize`
        :return: Node type of compute node.
        """
    instance_type = compute.find('INSTANCE_TYPE')
    try:
        return next((node_size for node_size in self.list_sizes() if node_size.name == instance_type.text))
    except StopIteration:
        return None