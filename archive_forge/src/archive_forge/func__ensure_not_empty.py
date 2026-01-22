import copy
import urllib
from oslo_serialization import jsonutils
from keystoneauth1 import exceptions
from mistralclient import utils
def _ensure_not_empty(self, **kwargs):
    for name, value in kwargs.items():
        if value is None or (isinstance(value, str) and len(value) == 0):
            raise APIException(400, '%s is missing field "%s"' % (self.resource_class.__name__, name))