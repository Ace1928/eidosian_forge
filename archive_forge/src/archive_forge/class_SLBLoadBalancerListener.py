from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
class SLBLoadBalancerListener(ReprMixin):
    """
    Base SLB load balancer listener class
    """
    _repr_attributes = ['port', 'backend_port', 'scheduler', 'bandwidth']
    action = None
    option_keys = []

    def __init__(self, port, backend_port, algorithm, bandwidth, extra=None):
        self.port = port
        self.backend_port = backend_port
        self.scheduler = ALGORITHM_TO_SLB_SCHEDULER.get(algorithm, 'wrr')
        self.bandwidth = bandwidth
        self.extra = extra or {}

    @classmethod
    def create(cls, port, backend_port, algorithm, bandwidth, extra=None):
        return cls(port, backend_port, algorithm, bandwidth, extra=extra)

    def get_create_params(self):
        params = self.get_required_params()
        options = self.get_optional_params()
        options.update(params)
        return options

    def get_required_params(self):
        params = {'Action': self.action, 'ListenerPort': self.port, 'BackendServerPort': self.backend_port, 'Scheduler': self.scheduler, 'Bandwidth': self.bandwidth}
        return params

    def get_optional_params(self):
        options = {}
        for option in self.option_keys:
            if self.extra and option in self.extra:
                options[option] = self.extra[option]
        return options