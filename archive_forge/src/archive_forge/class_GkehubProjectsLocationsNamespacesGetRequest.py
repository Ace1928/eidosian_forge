from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsNamespacesGetRequest(_messages.Message):
    """A GkehubProjectsLocationsNamespacesGetRequest object.

  Fields:
    name: Required. The Namespace resource name in the format
      `projects/*/locations/*/namespaces/*`.
  """
    name = _messages.StringField(1, required=True)