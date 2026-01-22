from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunNamespacesServicesDeleteRequest(_messages.Message):
    """A RunNamespacesServicesDeleteRequest object.

  Fields:
    apiVersion: Not supported, and ignored by Cloud Run.
    dryRun: Indicates that the server should validate the request and populate
      default values without persisting the request. Supported values: `all`
    kind: Not supported, and ignored by Cloud Run.
    name: Required. The fully qualified name of the service to delete. It can
      be any of the following forms: *
      `namespaces/{project_id_or_number}/services/{service_name}` (only when
      the `endpoint` is regional) * `projects/{project_id_or_number}/locations
      /{region}/services/{service_name}` * `projects/{project_id_or_number}/re
      gions/{region}/services/{service_name}`
    propagationPolicy: Not supported, and ignored by Cloud Run.
  """
    apiVersion = _messages.StringField(1)
    dryRun = _messages.StringField(2)
    kind = _messages.StringField(3)
    name = _messages.StringField(4, required=True)
    propagationPolicy = _messages.StringField(5)