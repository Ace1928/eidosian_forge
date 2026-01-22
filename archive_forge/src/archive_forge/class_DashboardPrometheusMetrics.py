from typing import Optional
from ray.dashboard.consts import COMPONENT_METRICS_TAG_KEYS
class DashboardPrometheusMetrics(object):

    def __getattr__(self, attr):
        return NullMetric()