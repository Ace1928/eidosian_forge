from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsScopesNamespacesResourcequotasDeleteRequest(_messages.Message):
    """A GkehubProjectsLocationsScopesNamespacesResourcequotasDeleteRequest
  object.

  Fields:
    name: Required. The ResourceQuota resource name in the format
      `projects/*/locations/*/scopes/*/namespaces/*/resourcequotas/*`.
  """
    name = _messages.StringField(1, required=True)