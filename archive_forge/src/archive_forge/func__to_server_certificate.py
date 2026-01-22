from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def _to_server_certificate(self, el):
    _id = findtext(el, 'ServerCertificateId', namespace=self.namespace)
    name = findtext(el, 'ServerCertificateName', namespace=self.namespace)
    fingerprint = findtext(el, 'Fingerprint', namespace=self.namespace)
    return SLBServerCertificate(id=_id, name=name, fingerprint=fingerprint)