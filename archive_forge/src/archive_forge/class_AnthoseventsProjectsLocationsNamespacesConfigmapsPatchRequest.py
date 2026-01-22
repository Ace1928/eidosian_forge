from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsProjectsLocationsNamespacesConfigmapsPatchRequest(_messages.Message):
    """A AnthoseventsProjectsLocationsNamespacesConfigmapsPatchRequest object.

  Fields:
    configMap: A ConfigMap resource to be passed as the request body.
    name: The name of the configMap being retrieved. If needed, replace
      {namespace_id} with the project ID.
  """
    configMap = _messages.MessageField('ConfigMap', 1)
    name = _messages.StringField(2, required=True)