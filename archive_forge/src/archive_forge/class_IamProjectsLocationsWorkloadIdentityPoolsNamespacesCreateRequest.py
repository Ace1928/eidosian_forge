from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsWorkloadIdentityPoolsNamespacesCreateRequest(_messages.Message):
    """A IamProjectsLocationsWorkloadIdentityPoolsNamespacesCreateRequest
  object.

  Fields:
    parent: Required. The parent resource to create the namespace in. The only
      supported location is `global`.
    workloadIdentityPoolNamespace: A WorkloadIdentityPoolNamespace resource to
      be passed as the request body.
    workloadIdentityPoolNamespaceId: Required. The ID to use for the
      namespace. This value must: * contain at most 63 characters * contain
      only lowercase alphanumeric characters or `-` * start with an
      alphanumeric character * end with an alphanumeric character The prefix
      "gcp-" will be reserved for future uses.
  """
    parent = _messages.StringField(1, required=True)
    workloadIdentityPoolNamespace = _messages.MessageField('WorkloadIdentityPoolNamespace', 2)
    workloadIdentityPoolNamespaceId = _messages.StringField(3)