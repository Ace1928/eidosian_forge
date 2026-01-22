from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class GithubInstallationsService(base_api.BaseApiService):
    """Service class for the github_installations resource."""
    _NAME = 'github_installations'

    def __init__(self, client):
        super(CloudbuildV1.GithubInstallationsService, self).__init__(client)
        self._upload_configs = {}