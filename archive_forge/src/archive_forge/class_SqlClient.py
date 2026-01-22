from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import resources
class SqlClient(object):
    """Wrapper for SQL API client and associated resources."""

    def __init__(self, api_version):
        self.sql_client = apis.GetClientInstance('sql', api_version)
        self.sql_messages = self.sql_client.MESSAGES_MODULE
        self.resource_parser = resources.Registry()
        self.resource_parser.RegisterApiByName('sql', api_version)