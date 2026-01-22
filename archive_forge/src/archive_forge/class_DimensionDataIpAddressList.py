from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataIpAddressList:
    """
    DimensionData IP Address list
    """

    def __init__(self, id, name, description, ip_version, ip_address_collection, state, create_time, child_ip_address_lists=None):
        """"
        Initialize an instance of :class:`DimensionDataIpAddressList`

        :param id: GUID of the IP Address List key
        :type  id: ``str``

        :param name: Name of the IP Address List
        :type  name: ``str``

        :param description: Description of the IP Address List
        :type  description: ``str``

        :param ip_version: IP version. E.g. IPV4, IPV6
        :type  ip_version: ``str``

        :param ip_address_collection: Collection of DimensionDataIpAddress
        :type  ip_address_collection: ``List``

        :param state: IP Address list state
        :type  state: ``str``

        :param create_time: IP Address List created time
        :type  create_time: ``date time``

        :param child_ip_address_lists: List of IP address list to be included
        :type  child_ip_address_lists: List         of :class:'DimensionDataIpAddressList'
        """
        self.id = id
        self.name = name
        self.description = description
        self.ip_version = ip_version
        self.ip_address_collection = ip_address_collection
        self.state = state
        self.create_time = create_time
        self.child_ip_address_lists = child_ip_address_lists

    def __repr__(self):
        return '<DimensionDataIpAddressList: id=%s, name=%s, description=%s, ip_version=%s, ip_address_collection=%s, state=%s, create_time=%s, child_ip_address_lists=%s>' % (self.id, self.name, self.description, self.ip_version, self.ip_address_collection, self.state, self.create_time, self.child_ip_address_lists)