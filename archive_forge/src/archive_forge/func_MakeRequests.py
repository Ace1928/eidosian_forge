from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import cdn_flags_utils as cdn_flags
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute import signed_url_flags
from googlecloudsdk.command_lib.compute.backend_buckets import backend_buckets_utils
from googlecloudsdk.command_lib.compute.backend_buckets import flags as backend_buckets_flags
from googlecloudsdk.command_lib.compute.security_policies import (
from googlecloudsdk.core import log
def MakeRequests(self, args):
    """Makes the requests for updating the backend bucket."""
    holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
    client = holder.client
    backend_bucket_ref = self.BACKEND_BUCKET_ARG.ResolveAsResource(args, holder.resources)
    get_request = self.GetGetRequest(client, backend_bucket_ref)
    objects = client.MakeRequests([get_request])
    new_object, cleared_fields = self.Modify(args, objects[0])
    if objects[0] == new_object:
        if getattr(args, 'edge_security_policy', None) is None:
            log.status.Print('No change requested; skipping update for [{0}].'.format(objects[0].name))
            return objects
        backend_bucket_result = []
    else:
        with client.apitools_client.IncludeFields(cleared_fields):
            backend_bucket_result = client.MakeRequests([self.GetSetRequest(client, backend_bucket_ref, new_object)])
    if getattr(args, 'edge_security_policy', None) is not None:
        if getattr(args, 'edge_security_policy', None):
            security_policy_ref = self.EDGE_SECURITY_POLICY_ARG.ResolveAsResource(args, holder.resources).SelfLink()
        else:
            security_policy_ref = None
        edge_security_policy_request = self.GetSetEdgeSecurityPolicyRequest(client, backend_bucket_ref, security_policy_ref)
        edge_security_policy_result = client.MakeRequests([edge_security_policy_request])
    else:
        edge_security_policy_result = []
    return backend_bucket_result + edge_security_policy_result