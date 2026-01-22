from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataPoolMember:
    """
    DimensionData VIP Pool Member.
    """

    def __init__(self, id, name, status, ip, port, node_id):
        """
        Initialize an instance of ``DimensionDataPoolMember``

        :param id: The ID of the pool member
        :type  id: ``str``

        :param name: The name of the pool member
        :type  name: ``str``

        :param status: The status of the pool
        :type  status: :class:`DimensionDataStatus`

        :param ip: The IP of the pool member
        :type  ip: ``str``

        :param port: The port of the pool member
        :type  port: ``int``

        :param node_id: The ID of the associated node
        :type  node_id: ``str``
        """
        self.id = str(id)
        self.name = name
        self.status = status
        self.ip = ip
        self.port = port
        self.node_id = node_id

    def __repr__(self):
        return '<DimensionDataPoolMember: id=%s, name=%s, ip=%s, status=%s, port=%s, node_id=%s>' % (self.id, self.name, self.ip, self.status, self.port, self.node_id)