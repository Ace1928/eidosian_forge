from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsScopesNamespacesGetRequest(_messages.Message):
    """A GkehubProjectsLocationsScopesNamespacesGetRequest object.

  Fields:
    name: Required. The Namespace resource name in the format
      `projects/*/locations/*/scopes/*/namespaces/*`.
  """
    name = _messages.StringField(1, required=True)