from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def _to_listeners(self, data):
    xpath = 'DescribeListenersResult/Listeners/member'
    return [self._to_listener(el) for el in findall(element=data, xpath=xpath, namespace=NS)]