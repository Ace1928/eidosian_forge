from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core import exceptions
class RegionServerTlsPoliciesCompleter(completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(RegionServerTlsPoliciesCompleter, self).__init__(collection='networksecurity.projects.locations.serverTlsPolicies', api_version='v1alpha1', list_command='network-security server-tls-policies list --filter=region:* --uri', **kwargs)