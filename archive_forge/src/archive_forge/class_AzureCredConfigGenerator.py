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
class AzureCredConfigGenerator(CredConfigGenerator):
    """The generator for Azure-based credential configs."""

    def __init__(self, app_id_uri, audience):
        super(AzureCredConfigGenerator, self).__init__(ConfigType.WORKLOAD_IDENTITY_POOLS)
        self.app_id_uri = app_id_uri
        self.audience = audience

    def get_token_type(self, subject_token_type):
        return 'urn:ietf:params:oauth:token-type:jwt'

    def get_source(self, args):
        self._format_already_defined(args.credential_source_type)
        return {'url': 'http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=' + (self.app_id_uri or 'https://iam.googleapis.com/' + self.audience), 'headers': {'Metadata': 'True'}, 'format': {'type': 'json', 'subject_token_field_name': 'access_token'}}