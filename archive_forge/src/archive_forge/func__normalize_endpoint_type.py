import abc
import warnings
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _normalize_endpoint_type(self, endpoint_type):
    if endpoint_type:
        endpoint_type = endpoint_type.rstrip('URL')
    return endpoint_type