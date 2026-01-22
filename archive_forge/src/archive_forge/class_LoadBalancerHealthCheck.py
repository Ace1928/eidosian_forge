from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal
from ..core import BaseDomain
class LoadBalancerHealthCheck(BaseDomain):
    """LoadBalancerHealthCheck Domain

    :param protocol: str
        Protocol of the service Choices: tcp, http, https
    :param port: int
        Port the healthcheck will be performed on
    :param interval: int
           Interval we trigger health check in
    :param timeout: int
            Timeout in sec after a try is assumed as timeout
    :param retries: int
            Retries we perform until we assume a target as unhealthy
    :param http: LoadBalancerHealtCheckHttp
            HTTP Config
    """

    def __init__(self, protocol: str | None=None, port: int | None=None, interval: int | None=None, timeout: int | None=None, retries: int | None=None, http: LoadBalancerHealtCheckHttp | None=None):
        self.protocol = protocol
        self.port = port
        self.interval = interval
        self.timeout = timeout
        self.retries = retries
        self.http = http