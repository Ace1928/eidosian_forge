from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import re
from apitools.base.py import list_pager
from cloudsdk.google.protobuf import timestamp_pb2
from googlecloudsdk.api_lib.spanner import response_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def SetPolicy(instance_ref, policy, field_mask=None):
    """Saves the given policy on the instance, overwriting whatever exists."""
    client = apis.GetClientInstance(_SPANNER_API_NAME, _SPANNER_API_VERSION)
    msgs = apis.GetMessagesModule(_SPANNER_API_NAME, _SPANNER_API_VERSION)
    policy.version = iam_util.MAX_LIBRARY_IAM_SUPPORTED_VERSION
    req = msgs.SpannerProjectsInstancesSetIamPolicyRequest(resource=instance_ref.RelativeName(), setIamPolicyRequest=msgs.SetIamPolicyRequest(policy=policy, updateMask=field_mask))
    return client.projects_instances.SetIamPolicy(req)