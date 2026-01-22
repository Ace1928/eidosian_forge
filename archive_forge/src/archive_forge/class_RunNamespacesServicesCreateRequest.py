from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunNamespacesServicesCreateRequest(_messages.Message):
    """A RunNamespacesServicesCreateRequest object.

  Fields:
    dryRun: Indicates that the server should validate the request and populate
      default values without persisting the request. Supported values: `all`
    parent: Required. The resource's parent. In Cloud Run, it may be one of
      the following: * `{project_id_or_number}` *
      `namespaces/{project_id_or_number}` *
      `namespaces/{project_id_or_number}/services` *
      `projects/{project_id_or_number}/locations/{region}` *
      `projects/{project_id_or_number}/regions/{region}`
    service: A Service resource to be passed as the request body.
  """
    dryRun = _messages.StringField(1)
    parent = _messages.StringField(2, required=True)
    service = _messages.MessageField('Service', 3)