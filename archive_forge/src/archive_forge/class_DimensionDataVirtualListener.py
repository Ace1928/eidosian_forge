from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataVirtualListener:
    """
    DimensionData Virtual Listener.
    """

    def __init__(self, id, name, status, ip):
        """
        Initialize an instance of :class:`DimensionDataVirtualListener`

        :param id: The ID of the listener
        :type  id: ``str``

        :param name: The name of the listener
        :type  name: ``str``

        :param status: The status of the listener
        :type  status: :class:`DimensionDataStatus`

        :param ip: The IP of the listener
        :type  ip: ``str``
        """
        self.id = str(id)
        self.name = name
        self.status = status
        self.ip = ip

    def __repr__(self):
        return '<DimensionDataVirtualListener: id=%s, name=%s, status=%s, ip=%s>' % (self.id, self.name, self.status, self.ip)