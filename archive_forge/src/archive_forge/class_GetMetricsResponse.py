from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal
from ..core import BaseDomain
class GetMetricsResponse(BaseDomain):
    """Get a Load Balancer Metrics Response Domain

    :param metrics: The Load Balancer metrics
    """
    __slots__ = ('metrics',)

    def __init__(self, metrics: Metrics):
        self.metrics = metrics