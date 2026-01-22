from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlloydbProjectsLocationsClustersInstancesRestartRequest(_messages.Message):
    """A AlloydbProjectsLocationsClustersInstancesRestartRequest object.

  Fields:
    name: Required. The name of the resource. For the required format, see the
      comment on the Instance.name field.
    restartInstanceRequest: A RestartInstanceRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    restartInstanceRequest = _messages.MessageField('RestartInstanceRequest', 2)