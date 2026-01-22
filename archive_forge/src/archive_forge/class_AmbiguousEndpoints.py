import inspect
import sys
from magnumclient.i18n import _
class AmbiguousEndpoints(EndpointException):
    """Found more than one matching endpoint in Service Catalog."""

    def __init__(self, endpoints=None):
        super(AmbiguousEndpoints, self).__init__(_('AmbiguousEndpoints: %r') % endpoints)
        self.endpoints = endpoints