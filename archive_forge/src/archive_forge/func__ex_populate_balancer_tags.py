from libcloud.utils.py3 import httplib
from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def _ex_populate_balancer_tags(self, balancer):
    tags = balancer.extra.get('tags', {})
    tags.update(self._ex_list_balancer_tags(balancer.id))
    if tags:
        balancer.extra['tags'] = tags
    return balancer