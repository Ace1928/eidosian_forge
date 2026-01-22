from magnumclient.common.apiclient import exceptions
from magnumclient.common.apiclient.exceptions import *  # noqa
class AmbiguousAuthSystem(exceptions.ClientException):
    """Could not obtain token and endpoint using provided credentials."""
    pass