from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesDemoteMasterRequest(_messages.Message):
    """A SqlInstancesDemoteMasterRequest object.

  Fields:
    instance: Cloud SQL instance name.
    instancesDemoteMasterRequest: A InstancesDemoteMasterRequest resource to
      be passed as the request body.
    project: ID of the project that contains the instance.
  """
    instance = _messages.StringField(1, required=True)
    instancesDemoteMasterRequest = _messages.MessageField('InstancesDemoteMasterRequest', 2)
    project = _messages.StringField(3, required=True)