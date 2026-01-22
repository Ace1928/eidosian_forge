from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core.console import console_io
def DoSetIamPolicy(instance_ref, namespace, new_iam_policy, messages, client):
    """Sets IAM policy for a given instance or a namespace."""
    if namespace:
        policy_request = messages.DatafusionProjectsLocationsInstancesNamespacesSetIamPolicyRequest(resource='%s/namespaces/%s' % (instance_ref.RelativeName(), namespace), setIamPolicyRequest=messages.SetIamPolicyRequest(policy=new_iam_policy))
        return client.projects_locations_instances_namespaces.SetIamPolicy(policy_request)
    else:
        policy_request = messages.DatafusionProjectsLocationsInstancesSetIamPolicyRequest(resource=instance_ref.RelativeName(), setIamPolicyRequest=messages.SetIamPolicyRequest(policy=new_iam_policy))
        return client.projects_locations_instances.SetIamPolicy(policy_request)