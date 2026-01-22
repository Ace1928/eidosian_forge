from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataAntiAffinityRule:
    """
    Anti-Affinity rule for DimensionData

    An Anti-Affinity rule ensures that servers in the rule will
    not reside on the same VMware ESX host.
    """

    def __init__(self, id, node_list):
        """
        Instantiate a new :class:`DimensionDataAntiAffinityRule`

        :param id: The ID of the Anti-Affinity rule
        :type  id: ``str``

        :param node_list: List of node ids that belong in this rule
        :type  node_list: ``list`` of ``str``
        """
        self.id = id
        self.node_list = node_list

    def __repr__(self):
        return '<DimensionDataAntiAffinityRule: id=%s>' % self.id