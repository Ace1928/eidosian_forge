from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import iam_util
class SecretsLatest(Client):
    """High-level client for latest secrets."""

    def __init__(self, client=None, messages=None, api_versions=None):
        super(SecretsLatest, self).__init__(client, messages, api_versions)
        self.service = self.client.projects_secrets_latest

    def Access(self, secret_ref, secret_location=None):
        """Access the latest version of a secret."""
        return self.service.Access(self.messages.SecretmanagerProjectsSecretsLatestAccessRequest(name=GetRelativeName(secret_ref, secret_location)))