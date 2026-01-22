from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
class SLBLoadBalancerHttpsListener(SLBLoadBalancerListener):
    """
    This class represents a rule to route https request to the backends.
    """
    action = 'CreateLoadBalancerHTTPSListener'
    option_keys = ['XForwardedFor', 'StickySessionType', 'CookieTimeout', 'Cookie', 'HealthCheckDomain', 'HealthCheckURI', 'HealthCheckConnectPort', 'HealthyThreshold', 'UnhealthyThreshold', 'HealthCheckTimeout', 'HealthCheckInterval', 'HealthCheckHttpCode']

    def __init__(self, port, backend_port, algorithm, bandwidth, sticky_session, health_check, certificate_id, extra=None):
        super().__init__(port, backend_port, algorithm, bandwidth, extra=extra)
        self.sticky_session = sticky_session
        self.health_check = health_check
        self.certificate_id = certificate_id

    def get_required_params(self):
        params = super().get_required_params()
        params['StickySession'] = self.sticky_session
        params['HealthCheck'] = self.health_check
        params['ServerCertificateId'] = self.certificate_id
        return params

    @classmethod
    def create(cls, port, backend_port, algorithm, bandwidth, extra={}):
        if 'StickySession' not in extra:
            raise AttributeError('StickySession is required')
        if 'HealthCheck' not in extra:
            raise AttributeError('HealthCheck is required')
        if 'ServerCertificateId' not in extra:
            raise AttributeError('ServerCertificateId is required')
        sticky_session = extra['StickySession']
        health_check = extra['HealthCheck']
        certificate_id = extra['ServerCertificateId']
        return cls(port, backend_port, algorithm, bandwidth, sticky_session, health_check, certificate_id, extra=extra)