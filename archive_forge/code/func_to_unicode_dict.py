from unittest import mock
from oslo_serialization import jsonutils
import sys
from keystoneauth1 import fixture
import requests
def to_unicode_dict(catalog_dict):
    """Converts dict to unicode dict

    """
    if isinstance(catalog_dict, dict):
        return {to_unicode_dict(key): to_unicode_dict(value) for key, value in catalog_dict.items()}
    elif isinstance(catalog_dict, list):
        return [to_unicode_dict(element) for element in catalog_dict]
    elif isinstance(catalog_dict, str):
        return catalog_dict + u''
    else:
        return catalog_dict