from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import json
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
class IamEndpoints(ByoidEndpoints):
    """Simple class to build IAM Credential endpoints."""

    def __init__(self, service_account, **kwargs):
        self._service_account = service_account
        super(IamEndpoints, self).__init__('iamcredentials', **kwargs)

    @property
    def impersonation_url(self):
        api = 'v1/projects/-/serviceAccounts/{}:generateAccessToken'.format(self._service_account)
        return '{}/{}'.format(self._base_url, api)