from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsProjectsLocationsConfigmapsCreateRequest(_messages.Message):
    """A AnthoseventsProjectsLocationsConfigmapsCreateRequest object.

  Fields:
    configMap: A ConfigMap resource to be passed as the request body.
    parent: Required. The namespace to create the configmap in
  """
    configMap = _messages.MessageField('ConfigMap', 1)
    parent = _messages.StringField(2, required=True)