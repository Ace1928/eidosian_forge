from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsGetIamPolicyRequest(_messages.Message):
    """A CloudkmsProjectsLocationsKeyRingsGetIamPolicyRequest object.

  Fields:
    resource: REQUIRED: The resource for which the policy is being requested.
      See the operation documentation for the appropriate value for this
      field.
  """
    resource = _messages.StringField(1, required=True)