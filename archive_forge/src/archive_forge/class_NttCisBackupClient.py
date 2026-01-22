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
class NttCisBackupClient:
    """
    An object that represents a backup client
    """

    def __init__(self, id, type, status, schedule_policy, storage_policy, download_url, alert=None, running_job=None):
        """
        Initialize an instance of this class.

        :param id: Unique ID for the client
        :type  id: ``str``
        :param type: The type of client that this client is
        :type  type: :class:`NttCisBackupClientType`
        :param status: The states of this particular backup client.
                       i.e. (Unregistered)
        :type  status: ``str``
        :param schedule_policy: The schedule policy for this client
                                NOTE: NTTCIS only sends back the name
                                of the schedule policy, no further details
        :type  schedule_policy: ``str``
        :param storage_policy: The storage policy for this client
                               NOTE: NTTCIS only sends back the name
                               of the storage policy, no further details
        :type  storage_policy: ``str``
        :param download_url: The download url for this client
        :type  download_url: ``str``
        :param alert: The alert configured for this backup client (optional)
        :type  alert: :class:`NttCisBackupClientAlert`
        :param alert: The running job for the client (optional)
        :type  alert: :class:`NttCisBackupClientRunningJob`
        """
        self.id = id
        self.type = type
        self.status = status
        self.schedule_policy = schedule_policy
        self.storage_policy = storage_policy
        self.download_url = download_url
        self.alert = alert
        self.running_job = running_job

    def __repr__(self):
        return '<NttCisBackupClient: id=%s>' % self.id