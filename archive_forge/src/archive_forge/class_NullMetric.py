from typing import Optional
class NullMetric:
    """Mock metric class to be used in case of prometheus_client import error."""

    def set(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass

    def inc(self, *args, **kwargs):
        pass

    def labels(self, *args, **kwargs):
        return self

    def clear(self):
        pass