import abc
import contextlib
import hashlib
import os
from cinderclient.apiclient import base as common_base
from cinderclient import exceptions
from cinderclient import utils
def _get_with_base_url(self, url, response_key=None):
    resp, body = self.api.client.get_with_base_url(url)
    if response_key:
        return [self.resource_class(self, res, loaded=True) for res in body[response_key] if res]
    else:
        return self.resource_class(self, body, loaded=True)