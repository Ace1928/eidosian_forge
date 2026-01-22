from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core import exceptions
class ServiceLoadBalancingPoliciesCompleter(completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(ServiceLoadBalancingPoliciesCompleter, self).__init__(collection='networkservices.projects.locations.serviceLbPolicies', api_version='v1alpha1', list_command='network-services service-lb-policies list --location=global --uri', **kwargs)