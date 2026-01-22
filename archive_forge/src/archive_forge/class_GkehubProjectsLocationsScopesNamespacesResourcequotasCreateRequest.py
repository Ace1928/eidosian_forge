from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsScopesNamespacesResourcequotasCreateRequest(_messages.Message):
    """A GkehubProjectsLocationsScopesNamespacesResourcequotasCreateRequest
  object.

  Fields:
    parent: Required. The parent (project and location) where the
      ResourceQuota will be created. Specified in the format
      `projects/*/locations/*/scopes/*/namespaces/*`.
    resourceQuota: A ResourceQuota resource to be passed as the request body.
    resourceQuotaId: Required. Client chosen ID for the ResourceQuota.
      `resource_quota_id` must be a valid RFC 1123 compliant DNS label: 1. At
      most 63 characters in length 2. It must consist of lower case
      alphanumeric characters or `-` 3. It must start and end with an
      alphanumeric character Which can be expressed as the regex:
      `[a-z0-9]([-a-z0-9]*[a-z0-9])?`, with a maximum length of 63 characters.
  """
    parent = _messages.StringField(1, required=True)
    resourceQuota = _messages.MessageField('ResourceQuota', 2)
    resourceQuotaId = _messages.StringField(3)