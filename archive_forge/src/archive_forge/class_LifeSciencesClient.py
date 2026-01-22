from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
class LifeSciencesClient(object):
    """Client for calling Cloud Life Sciences APIs."""

    def __init__(self):
        super(LifeSciencesClient, self).__init__()
        self._api_version = 'v2beta'
        self._client = None
        self._resources = None

    @property
    def client(self):
        if self._client is None:
            self._client = apis.GetClientInstance('lifesciences', self._api_version)
        return self._client

    @property
    def messages(self):
        return self.client.MESSAGES_MODULE