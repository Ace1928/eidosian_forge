from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1beta import osconfig_v1beta_messages as messages
Lookup the effective guest policy that applies to a VM instance. This lookup merges all policies that are assigned to the instance ancestry.

      Args:
        request: (OsconfigProjectsZonesInstancesLookupEffectiveGuestPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EffectiveGuestPolicy) The response message.
      