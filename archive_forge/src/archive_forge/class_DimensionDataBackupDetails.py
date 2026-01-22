from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataBackupDetails:
    """
    Dimension Data Backup Details represents information about
    a targets backups configuration
    """

    def __init__(self, asset_id, service_plan, status, clients=None):
        """
        Initialize an instance of :class:`DimensionDataBackupDetails`

        :param asset_id: Asset identification for backups
        :type  asset_id: ``str``

        :param service_plan: The service plan for backups. i.e (Essentials)
        :type  service_plan: ``str``

        :param status: The overall status this backup target.
                       i.e. (unregistered)
        :type  status: ``str``

        :param clients: Backup clients attached to this target
        :type  clients: ``list`` of :class:`DimensionDataBackupClient`
        """
        self.asset_id = asset_id
        self.service_plan = service_plan
        self.status = status
        self.clients = clients

    def __repr__(self):
        return '<DimensionDataBackupDetails: id=%s>' % self.asset_id