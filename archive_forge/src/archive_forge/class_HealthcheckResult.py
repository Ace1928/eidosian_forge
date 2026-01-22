import abc
class HealthcheckResult(object):
    """Result of a ``healthcheck`` method call should be this object."""

    def __init__(self, available, reason, details=None):
        self.available = available
        self.reason = reason
        self.details = details