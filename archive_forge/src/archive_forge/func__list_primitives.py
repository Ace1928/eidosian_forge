import abc
import copy
from urllib import parse as urlparse
from ironicclient.common.apiclient import base
from ironicclient import exc
def _list_primitives(self, url, response_key=None, os_ironic_api_version=None, global_request_id=None):
    return self.__list(url, response_key=response_key, os_ironic_api_version=os_ironic_api_version, global_request_id=global_request_id)