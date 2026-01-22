from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def _create_list_params(self, params, items, label):
    """
        return parameter list
        """
    if isinstance(items, str):
        items = [items]
    for index, item in enumerate(items):
        params[label % (index + 1)] = item
    return params