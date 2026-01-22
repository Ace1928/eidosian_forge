from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataBackupClientRunningJob:
    """
    A running job for a given backup client
    """

    def __init__(self, id, status, percentage=0):
        """
        Initialize an instance of :class:`DimensionDataBackupClientRunningJob`

        :param id: The unique ID of the job
        :type  id: ``str``

        :param status: The status of the job i.e. Waiting
        :type  status: ``str``

        :param percentage: The percentage completion of the job
        :type  percentage: ``int``
        """
        self.id = id
        self.percentage = percentage
        self.status = status

    def __repr__(self):
        return '<DimensionDataBackupClientRunningJob: id=%s>' % self.id