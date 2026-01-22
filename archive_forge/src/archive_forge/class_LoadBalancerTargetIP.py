from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal
from ..core import BaseDomain
class LoadBalancerTargetIP(BaseDomain):
    """LoadBalancerTargetIP Domain

    :param ip: str Target IP
    """

    def __init__(self, ip: str | None=None):
        self.ip = ip