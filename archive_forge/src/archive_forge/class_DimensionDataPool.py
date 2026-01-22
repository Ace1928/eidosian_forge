from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataPool:
    """
    DimensionData VIP Pool.
    """

    def __init__(self, id, name, description, status, load_balance_method, health_monitor_id, service_down_action, slow_ramp_time):
        """
        Initialize an instance of ``DimensionDataPool``

        :param id: The ID of the pool
        :type  id: ``str``

        :param name: The name of the pool
        :type  name: ``str``

        :param description: Plan text description of the pool
        :type  description: ``str``

        :param status: The status of the pool
        :type  status: :class:`DimensionDataStatus`

        :param load_balance_method: The load balancer method
        :type  load_balance_method: ``str``

        :param health_monitor_id: The ID of the health monitor
        :type  health_monitor_id: ``str``

        :param service_down_action: Action to take when pool is down
        :type  service_down_action: ``str``

        :param slow_ramp_time: The ramp-up time for service recovery
        :type  slow_ramp_time: ``int``
        """
        self.id = str(id)
        self.name = name
        self.description = description
        self.status = status
        self.load_balance_method = load_balance_method
        self.health_monitor_id = health_monitor_id
        self.service_down_action = service_down_action
        self.slow_ramp_time = slow_ramp_time

    def __repr__(self):
        return '<DimensionDataPool: id=%s, name=%s, description=%s, status=%s>' % (self.id, self.name, self.description, self.status)