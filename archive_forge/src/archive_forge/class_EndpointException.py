import inspect
import sys
from magnumclient.i18n import _
class EndpointException(ClientException):
    """Something is rotten in Service Catalog."""
    pass