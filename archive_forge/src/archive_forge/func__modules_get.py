import os
from troveclient import base
from troveclient import common
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules as core_modules
from swiftclient import client as swift_client
def _modules_get(self, instance, from_guest=None, include_contents=None):
    url = '/instances/%s/modules' % base.getid(instance)
    query_strings = {}
    if from_guest is not None:
        query_strings['from_guest'] = from_guest
    if include_contents is not None:
        query_strings['include_contents'] = include_contents
    url = common.append_query_strings(url, **query_strings)
    resp, body = self.api.client.get(url)
    common.check_for_exceptions(resp, body, url)
    return [core_modules.Module(self, module, loaded=True) for module in body['modules']]