from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def _to_server_and_weight(self, el):
    server_id = findtext(el, 'ServerId', namespace=self.namespace)
    weight = findtext(el, 'Weight', namespace=self.namespace)
    return {'ServerId': server_id, 'Weight': weight}