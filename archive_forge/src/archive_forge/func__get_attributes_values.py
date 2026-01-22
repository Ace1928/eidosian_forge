import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
def _get_attributes_values(self, attributes, element):
    values = {}
    for attribute_name, attribute_type, alias in attributes:
        key = alias if alias else attribute_name.upper()
        value = element.findtext(key)
        if value is not None:
            value = attribute_type(value)
        values[attribute_name] = value
    return values