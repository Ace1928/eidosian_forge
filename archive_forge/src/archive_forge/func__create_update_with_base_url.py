import abc
import contextlib
import hashlib
import os
from cinderclient.apiclient import base as common_base
from cinderclient import exceptions
from cinderclient import utils
def _create_update_with_base_url(self, url, body, response_key=None):
    resp, body = self.api.client.create_update_with_base_url(url, body=body)
    if response_key:
        return self.resource_class(self, body[response_key], loaded=True)
    return self.resource_class(self, body, loaded=True)