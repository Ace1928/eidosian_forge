from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def _ex_get_balancer_listeners(self, balancer):
    """
        Return a list of listeners associated with load balancer.

        :param balancer: Load balancer to fetch listeners for
        :type balancer: :class:`LoadBalancer`

        :return: list of listener objects
        :rtype: ``list`` of :class:`ALBListener`
        """
    params = {'Action': 'DescribeListeners', 'LoadBalancerArn': balancer.id}
    data = self.connection.request(ROOT, params=params).object
    return self._to_listeners(data)