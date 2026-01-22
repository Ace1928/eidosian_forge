from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
class ALBListener:
    """
    AWS ALB listener class
    http://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-listeners.html
    """

    def __init__(self, listener_id, protocol, port, balancer, driver, action='', ssl_policy='', ssl_certificate='', rules=[]):
        self.id = listener_id
        self.protocol = protocol
        self.port = port
        self.action = action
        self.ssl_policy = ssl_policy
        self.ssl_certificate = ssl_certificate
        self._balancer = balancer
        self._balancer_arn = balancer.id if balancer else None
        self._rules = rules
        self._driver = driver

    @property
    def balancer(self):
        if not self._balancer and self._balancer_arn:
            self._balancer = self._driver.get_balancer(self._balancer_arn)
        return self._balancer

    @balancer.setter
    def balancer(self, val):
        self._balancer = val
        self._balancer_arn = val.id

    @property
    def rules(self):
        if not self._rules:
            self._rules = self._driver._ex_get_rules_for_listener(self)
        return self._rules

    @rules.setter
    def rules(self, val):
        self._rules = val