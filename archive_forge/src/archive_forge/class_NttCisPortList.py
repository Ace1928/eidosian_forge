import re
import xml.etree.ElementTree as etree
from io import BytesIO
from copy import deepcopy
from time import sleep
from base64 import b64encode
from typing import Dict
from functools import wraps
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class NttCisPortList:
    """
    NttCis Port list
    """

    def __init__(self, id, name, description, port_collection, child_portlist_list, state, create_time):
        """ "
        Initialize an instance of :class:`DNttCisPortList`

        :param id: GUID of the Port List key
        :type  id: ``str``

        :param name: Name of the Port List
        :type  name: ``str``

        :param description: Description of the Port List
        :type  description: ``str``

        :param port_collection: Collection of NttCisPort
        :type  port_collection: ``List``

        :param child_portlist_list: Collection of NttCisChildPort
        :type  child_portlist_list: ``List``

        :param state: Port list state
        :type  state: ``str``

        :param create_time: Port List created time
        :type  create_time: ``date time``
        """
        self.id = id
        self.name = name
        self.description = description
        self.port_collection = port_collection
        self.child_portlist_list = child_portlist_list
        self.state = state
        self.create_time = create_time

    def __repr__(self):
        return '<NttCisPortList: id=%s, name=%s, description=%s, port_collection=%s, child_portlist_list=%s, state=%s, create_time=%s>' % (self.id, self.name, self.description, self.port_collection, self.child_portlist_list, self.state, self.create_time)