from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyName(_messages.Message):
    """An internal name for an IAM policy, based on the resource to which the
  policy applies. Not to be confused with a resource's external full resource
  name. For more information on this distinction, see go/iam-full-resource-
  names.

  Fields:
    id: Identifies an instance of the type. ID format varies by type. The ID
      format is defined in the IAM .service file that defines the type, either
      in path_mapping or in a comment.
    region: For Cloud IAM: The location of the Policy. Must be empty or
      "global" for Policies owned by global IAM. Must name a region from
      prodspec/cloud-iam-cloudspec for Regional IAM Policies, see go/iam-
      faq#where-is-iam-currently-deployed. For Local IAM: This field should be
      set to "local".
    type: Resource type. Types are defined in IAM's .service files. Valid
      values for type might be 'gce', 'gcs', 'project', 'account' etc.
  """
    id = _messages.StringField(1)
    region = _messages.StringField(2)
    type = _messages.StringField(3)