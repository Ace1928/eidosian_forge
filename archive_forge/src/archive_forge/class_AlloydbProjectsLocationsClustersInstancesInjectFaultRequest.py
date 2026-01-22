from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlloydbProjectsLocationsClustersInstancesInjectFaultRequest(_messages.Message):
    """A AlloydbProjectsLocationsClustersInstancesInjectFaultRequest object.

  Fields:
    injectFaultRequest: A InjectFaultRequest resource to be passed as the
      request body.
    name: Required. The name of the resource. For the required format, see the
      comment on the Instance.name field.
  """
    injectFaultRequest = _messages.MessageField('InjectFaultRequest', 1)
    name = _messages.StringField(2, required=True)