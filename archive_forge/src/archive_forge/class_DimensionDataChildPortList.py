from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataChildPortList:
    """
    DimensionData Child Port list
    """

    def __init__(self, id, name):
        """ "
        Initialize an instance of :class:`DimensionDataChildIpAddressList`

        :param id: GUID of the child port list key
        :type  id: ``str``

        :param name: Name of the child port List
        :type  name: ``str``

        """
        self.id = id
        self.name = name

    def __repr__(self):
        return '<DimensionDataChildPortList: id={}, name={}>'.format(self.id, self.name)