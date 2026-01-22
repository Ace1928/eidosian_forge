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
class NttCisBackupStoragePolicy:
    """
    A representation of a storage policy
    """

    def __init__(self, name, retention_period, secondary_location):
        """
        Initialize an instance of :class:`NttCisBackupStoragePolicy`

        :param name: The name of the storage policy i.e. 14 Day Storage Policy
        :type  name: ``str``

        :param retention_period: How long to keep the backup in days
        :type  retention_period: ``int``

        :param secondary_location: The secondary location i.e. Primary
        :type  secondary_location: ``str``
        """
        self.name = name
        self.retention_period = retention_period
        self.secondary_location = secondary_location

    def __repr__(self):
        return '<NttCisBackupStoragePolicy: name=%s>' % self.name